import torch

torch_model_path = f"./wafer_yolov5s_best.pt"
onnx_model_path = "./weights/wafer_yolov5s_best.onnx"
IMAGE_SIZE = 640
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./wafer_yolov5s_best.pt')
model = torch.load(torch_model_path,map_location=torch.device('cpu'))

torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)