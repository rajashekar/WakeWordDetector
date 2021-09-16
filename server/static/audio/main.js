window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;
var socketio = io.connect(location.origin + '/audio');

const bufferSize = 1024
const channels = 1
const sampleRate = 16000

const wakeWords = ["hey", "fourth", "brain"]

let bufferMap = {}

addprediction = wordJson => {
    word = wordJson['word']
    if(wakeWords.filter(i => i == word).length) {
        addWordSummary(wordJson)
    } else {
        if (word.endsWith('detected')) {
            // play prompt
            let prompt = new Audio("static/audio/prompt.mp3")
            prompt.play()
            // stop recording
            document.getElementById("record").click();
        } 
        words = document.createElement('p');
        words.innerHTML = '<b>' + word + '</b>';
        document.getElementById('wavefiles').appendChild(words);
    }
}


const addWordSummary = function(wordJson) {
    word = wordJson['word']
    // add buffer
    addAudioBuffer(wordJson)
    // add images
    timePlot = new Image()
    timePlot.src = `data:image/jpg;base64,${wordJson['time']}`
    melPlot = new Image()
    melPlot.src = `data:image/jpg;base64,${wordJson['mel']}`
    logMelPlot = new Image()
    logMelPlot.src = `data:image/jpg;base64,${wordJson['logmel']}`

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


function playBuffer(buffer) {
    var audioBuffer = audioContext.createBuffer(1, buffer.length, sampleRate);
    var audioBufferData = audioBuffer.getChannelData(0);
    audioBufferData.set(buffer);
    var source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}


const addAudioBuffer = function(wordJson) {
    bufferMap[`${wordJson['word']}_buffer`] = wordJson['buffer']
    // create play button
    playbtn = document.createElement('button');
    playbtn.innerHTML = word
    playbtn.onclick = function() {
        playBuffer(bufferMap[`${this.innerText}_buffer`])
    }
    document.getElementById('wavefiles').appendChild(playbtn);
}


socketio.on('add-prediction', wordJson => addprediction(JSON.parse(wordJson)));

function toggleRecording( e ) {
    // Chrome is suspending the audio context on load
    if (audioContext.state == "suspended") {
        audioContext.resume()
    }
    if (e.classList.contains('recording')) {
        // stop recording
        e.classList.remove('recording');
        recording = false;
        socketio.emit('end-recording');
    } else {
        // start recording
        document.getElementById('wavefiles').innerHTML = ""
        addprediction(JSON.parse('{"word": "Listening for wake words [hey, fourth, brain] ..."}'))
        e.classList.add('recording');
        recording = true;
        socketio.emit('start-recording', {numChannels: channels, bufferSize: bufferSize, bps: 16, fps: parseInt(audioContext.sampleRate)});
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

function toggleMono() {
    if (audioInput != realAudioInput) {
        audioInput.disconnect();
        realAudioInput.disconnect();
        audioInput = realAudioInput;
    } else {
        realAudioInput.disconnect();
        audioInput = convertToMono( realAudioInput );
    }

    audioInput.connect(inputPoint);
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
    scriptNode.onaudioprocess = function (audioEvent) {
        if (recording) {
            input = audioEvent.inputBuffer.getChannelData(0);

            // convert float audio data to 16-bit PCM
            var buffer = new ArrayBuffer(input.length * 2)
            var output = new DataView(buffer);
            for (var i = 0, offset = 0; i < input.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            socketio.emit('write-audio', buffer);
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