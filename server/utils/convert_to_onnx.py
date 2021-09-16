import torch

from model import CNN


def main():
    wake_words = ["hey", "fourth", "brain"]
    num_labels = len(wake_words) + 1  # oov
    num_maps1 = 48
    num_maps2 = 64
    num_hidden_input = 768
    hidden_size = 128
    batch_size = 1

    # get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    pytorch_model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
    pytorch_model.load_state_dict(
        torch.load("trained_models/model_hey_fourth_brain_with_noise.pt", map_location=device)
    )
    # put in eval mode
    pytorch_model.eval()
    # define the input size
    input_size = (1, 40, 61)
    # generate dummy data
    dummy_input = torch.rand(batch_size, *input_size).type(torch.FloatTensor).to(device=device)
    # generate onnx file
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        "trained_models/onnx_model.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        verbose=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )


if __name__ == "__main__":
    main()
