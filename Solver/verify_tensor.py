#VERIFYMETRIC Verifies the metric tensor and stress energy tensor structs
from Metrics import Metric


def verify_tensor(input_tensor: Metric, suppress_msgs: bool = False):
    verified: bool = True

    if input_tensor.type is not None:
        if input_tensor.type is "Metric":
            if not suppress_msgs:
                print("Type: Metric")
        elif input_tensor.type is "Stress-Energy":
            if not suppress_msgs:
                print("Type: Stress-Energy")
        else:
            verified = False
            raise Warning("Metric type unknown")
