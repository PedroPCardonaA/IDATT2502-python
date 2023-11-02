
import torch
import torch.onnx
import onnx
import models

def main():
    model = models.garbage_classifier_5L_attention_with_batch_and_dropout(input_shape=3, hidden_units=64, output_shape=8)

    model.load_state_dict(torch.load("models/modelX03.pth", map_location=torch.device('cpu')))
    onnx_path = "models/model.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    main()