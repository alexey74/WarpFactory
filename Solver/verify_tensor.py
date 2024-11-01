#VERIFYMETRIC Verifies the metric tensor and stress energy tensor structs
from Metrics import Metric


# TODO: Metric case


def verify_tensor(input_tensor: Metric, suppress_msgs: bool = False):
    verified: bool = True

    if input_tensor.type is not None:
        if input_tensor.type == "metric":
            if not suppress_msgs:
                print("Type: Metric")
        elif input_tensor.type == "Stress-Energy":
            if not suppress_msgs:
                print("Type: Stress-Energy")
        else:
            verified = False
            raise Warning("Metric type unknown")
    return verified