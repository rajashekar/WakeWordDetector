window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

const classes = ["hey", "fourth", "brain", "oov"]
const wakeWords = ["hey", "fourth", "brain"]
const bufferSize = 1024
const channels = 1
const windowSize = 750
const zmuv_mean = 0.000016
const zmuv_std = 0.072771
const log_offset = 1e-7
const SPEC_HOP_LENGTH = 200;
const MEL_SPEC_BINS = 40;
const NUM_FFTS = 512;
const audioFloatSize = 32767
const sampleRate = 16000
const numOfBatches = 3

let predictWords = []
let arrayBuffer = []
let targetState = 0

let bufferMap = {}

const windowBufferSize = windowSize/1000 * sampleRate

let session;
async function loadModel() {
    session = new onnx.InferenceSession();
    await session.loadModel("static/audio/onnx_model.onnx");
}
loadModel()

const addprediction = function(word) {
    if(wakeWords.filter(i => i == word).length) {
        addWordSummary(word)
    } else {
        words = document.createElement('p');
        words.innerHTML = '<b>' + word + '</b>';
        document.getElementById('wavefiles').appendChild(words);
    }
}

function playBuffer(buffer) {
    var audioBuffer = audioContext.createBuffer(1, buffer.length, SAMPLE_RATE);
    var audioBufferData = audioBuffer.getChannelData(0);
    audioBufferData.set(buffer);
    var source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

function createLayout(title, xTitle, yTitle, params={}) {
    const logY = (params.logY == true);
    const logX = (params.logX == true);
    return {
      title: title,
      xaxis: {
        title: xTitle,
        type: logX ? 'log' : null,
      },
      yaxis: {
        title: yTitle,
        type: logY ? 'log' : null,
      }
    }
  }
 
  
function plotAudio(y, layout) {
    //let t = y.map((value, index) => (index / sr));
    let t = y.map((value, index) => index);
    return plotXY(t, y, layout);
}

function plotXY(x, y, layout) {
    const out = document.createElement('div');
    out.className = 'plot';
    const xArr = Array.prototype.slice.call(x);
    const yArr = Array.prototype.slice.call(y);
    const data = [{
      x: xArr,
      y: yArr,
    }]
    Plotly.newPlot(out, data, layout);
    return out;
}

function plotSpectrogram(spec, samplesPerSlice, layout) {
    return plotImage(spec, samplesPerSlice, layout);
}

function plotImage(stft, samplesPerSlice, layout) {
    let out = document.createElement('div');
    out.className = 'plot';
    // Transpose the spectrogram we pass in.
    let zArr = [];
    for (let i = 0; i < stft.length; i++) {
      for (let j = 0; j < stft[0].length; j++) {
        if (zArr[j] == undefined) {
          zArr[j] = [];
        }
        // librosa.power_to_db(spec)
        zArr[j][i] = 10 * Math.log10(stft[i][j]) - 10 * Math.log10(1)
      }
    }
    // Calculate the X values (times) from the stft params.
    //const xArr = stft.map((value, index) => index * samplesPerSlice / sr);
    const xArr = stft.map((value, index) => index);
    // Calculate Y values (frequencies) from stft.
    const fft = Array.prototype.slice.call(stft[0]);
    const yArr = fft.map((value, index) => index);
  
    const data = [
      {
        x: xArr,
        y: yArr,
        z: zArr,
        type: 'heatmap'
      }
    ];
    Plotly.newPlot(out, data, layout);
    return out;
  }



const addWordSummary = function(word) {
    // create play button
    playbtn = document.createElement('button');
    playbtn.innerHTML = word
    playbtn.onclick = function() {
        playBuffer(bufferMap[`${this.innerText}_buffer`])
    }

    let arrayBuffer = bufferMap[`${word}_buffer`]
    let log_mels = bufferMap[`${word}_mels`]

    timePlot = plotAudio(arrayBuffer, createLayout('Time domain', 'Time (samples)', 'Amplitude'));
    melPlot = plotSpectrogram(log_mels, SPEC_HOP_LENGTH, createLayout('Melspectrogram', 'frame', 'mel freq'));

    // compute log_mels
    let log_mels_offset = [];
    for (let i = 0; i < log_mels.length; i++) {
        for (let j = 0; j < log_mels[0].length; j++) {
        if (log_mels_offset[i] == undefined) {
            log_mels_offset[i] = [];
        }
        log_mels_offset[i][j] = Math.log(log_mels[i][j] + log_offset)
        }
    }

    logMelPlot = plotSpectrogram(log_mels_offset, SPEC_HOP_LENGTH, createLayout('Log Melspectrogram', 'frame', 'mel freq'));

    document.getElementById('wavefiles').appendChild(playbtn);
    rowDiv = document.createElement('div');
    rowDiv.classList.add("row")

    colDiv = document.createElement('div')
    colDiv.classList.add("column")
    colDiv.appendChild(timePlot)
    rowDiv.appendChild(colDiv)

    colDiv = document.createElement('div')
    colDiv.classList.add("column")
    colDiv.appendChild(melPlot)
    rowDiv.appendChild(colDiv)

    colDiv = document.createElement('div')
    colDiv.classList.add("column")
    colDiv.appendChild(logMelPlot)
    rowDiv.appendChild(colDiv)

    document.getElementById('wavefiles').appendChild(rowDiv);
}

function toggleRecording( e ) {
    // Chrome is suspending the audio context on load
    if (audioContext.state == "suspended") {
        audioContext.resume()
    }
    if (e.classList.contains('recording')) {
        // stop recording
        e.classList.remove('recording');
        recording = false;
    } else {
        // start recording
        document.getElementById('wavefiles').innerHTML = ""
        addprediction('Listening for wake words [hey, fourth, brain] ...')
        e.classList.add('recording');
        recording = true;
    }
}

function convertToMono( input ) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect( splitter );
    splitter.connect( merger, 0, 0 );
    splitter.connect( merger, 0, 1 );
    return merger;
}

function cancelAnalyserUpdates() {
    window.cancelAnimationFrame( rafID );
    rafID = null;
}

function updateAnalysers(time) {
    if (!analyserContext) {
        var canvas = document.getElementById('analyser');
        canvasWidth = canvas.width;
        canvasHeight = canvas.height;
        analyserContext = canvas.getContext('2d');
    }

    // analyzer draw code here
    {
        var SPACING = 3;
        var BAR_WIDTH = 1;
        var numBars = Math.round(canvasWidth / SPACING);
        var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

        analyserNode.getByteFrequencyData(freqByteData); 

        analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
        analyserContext.fillStyle = '#F6D565';
        analyserContext.lineCap = 'round';
        var multiplier = analyserNode.frequencyBinCount / numBars;

        // Draw rectangle for each frequency bin.
        for (var i = 0; i < numBars; ++i) {
            var magnitude = 0;
            var offset = Math.floor( i * multiplier );
            // gotta sum/average the block, or we miss narrow-bandwidth spikes
            for (var j = 0; j< multiplier; j++)
                magnitude += freqByteData[offset + j];
            magnitude = magnitude / multiplier;
            var magnitude2 = freqByteData[i * multiplier];
            analyserContext.fillStyle = "hsl( " + Math.round((i*360)/numBars) + ", 100%, 50%)";
            analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
        }
    }
    
    rafID = window.requestAnimationFrame( updateAnalysers );
}

function flatten(log_mels) {
    flatten_arry = []
    for(i = 0; i < MEL_SPEC_BINS; i++) {
        for(j = 0; j < log_mels.length; j++) {
            flatten_arry.push((Math.log(log_mels[j][i] + log_offset) - zmuv_mean) / zmuv_std)
        }
    }
    return flatten_arry
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

const padArray = function(arr,len,fill) {
    return arr.concat(Array(len).fill(fill)).slice(0,len);
 }

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    // bufferSize, in_channels, out_channels
    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, bufferSize, channels, channels);
    scriptNode.onaudioprocess = async function (audioEvent) {
        if (recording) {
            let resampledMonoAudio = await resampleAndMakeMono(audioEvent.inputBuffer);
            arrayBuffer = [...arrayBuffer, ...resampledMonoAudio]
            batchSize = Math.floor(arrayBuffer.length/windowBufferSize)
            // if we got batches * 750 ms seconds of buffer 
            let batchBuffers = []
            let batchMels = []
            if (arrayBuffer.length >= numOfBatches * windowBufferSize) {
                let batch = 0
                let dataProcessed, log_mels;
                for (let i = 0; i < arrayBuffer.length; i = i + windowBufferSize) {
                    batchBuffer = arrayBuffer.slice(i, i+windowBufferSize)
                    //  if it is less than 750 ms then pad it with ones
                    if (batchBuffer.length < windowBufferSize) {
                        //batchBuffer = padArray(batchBuffer, windowBufferSize, 1)
                        // discard last slice
                        break
                    }
                    // arrayBuffer = arrayBuffer.filter(x => x/audioFloatSize)
                    // calculate log mels
                    log_mels = melSpectrogram(batchBuffer, {
                        sampleRate: sampleRate,
                        hopLength: SPEC_HOP_LENGTH,
                        nMels: MEL_SPEC_BINS,
                        nFft: NUM_FFTS
                    });
                    batchBuffers.push(batchBuffer)
                    batchMels.push(log_mels)
                    if (batch == 0) {
                        dataProcessed = []
                    }
                    dataProcessed = [...dataProcessed, ...flatten(log_mels)]
                    batch = batch + 1
                }
                // clear buffer
                arrayBuffer = []
                let inputTensor = new onnx.Tensor(dataProcessed, 'float32', [batch, 1, MEL_SPEC_BINS, dataProcessed.length/(batch * MEL_SPEC_BINS)]);
                // Run model with Tensor inputs and get the result.
                let outputMap = await session.run([inputTensor]);
                let outputData = outputMap.values().next().value.data;
                for (let i = 0; i<outputData.length; i = i+classes.length) {
                    let scores = Array.from(outputData.slice(i,i+classes.length))
                    console.log("scores", scores)
                    let probs = softmax(scores)
                    probs_sum = probs.reduce( (sum, x) => x+sum)
                    probs = probs.filter(x => x/probs_sum)
                    let class_idx = argMax(probs)
                    console.log("probabilities", probs)
                    console.log("predicted word", classes[class_idx])
                    if (classes[targetState] == classes[class_idx]) {
                        bufferMap[`${classes[targetState]}_buffer`] = batchBuffers[Math.floor(i/classes.length)]
                        bufferMap[`${classes[targetState]}_mels`] = batchMels[Math.floor(i/classes.length)]
                        console.log(classes[class_idx])
                        addprediction(classes[class_idx])
                        predictWords.push(classes[class_idx]) 
                        targetState += 1
                        if (wakeWords.join(' ') == predictWords.join(' ')) {
                            addprediction(`Wake word detected - ${predictWords.join(' ')}`)
                            let prompt = new Audio("static/audio/prompt.mp3");
                            prompt.play()
                            // stop recording
                            document.getElementById("record").click();
                            predictWords = []
                            targetState = 0
                        }
                    }
                }
            }
        }
    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    updateAnalysers();
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    // Chrome is suspending the audio context on load
    if (audioContext.state == "suspended") {
        audioContext.resume()
    }
    constraints = {audio: true}
    navigator.mediaDevices.getUserMedia(constraints)
        .then(gotStream)
        .catch(function(err) {
            alert('Error getting audio');
            console.log(err);
        });
}

window.addEventListener('load', initAudio );