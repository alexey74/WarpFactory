#VERIFYMETRIC Verifies the metric tensor and stress energy tensor structs


# TODO: Test


def verify_tensor(input_tensor, suppress_msgs: bool = False):
    verified: bool = True

    if hasattr(input_tensor, 'type') and input_tensor.type is not None:
        if input_tensor.type.casefold() == "metric":
            if not suppress_msgs:
                print("Type: Metric")
        elif input_tensor.type.casefold() == "stress-energy":
            if not suppress_msgs:
                print("Type: Stress-Energy")
        else:
            verified = False
            raise Warning("Metric type unknown")

        # Check other properties
        # Tensor
        if hasattr(input_tensor, 'tensor') and input_tensor.tensor is not None:
            size: tuple = input_tensor.tensor.shape
            if size[0] == 4 and size[1] == 4 and input_tensor.tensor.ndims - 2 == 4:
                if not suppress_msgs:
                    print("Tensor: Verified")
            else:
                verified = False
                raise Warning("Tensor is not formatted correctly. Tensor must be a 4x4 cell array of 4D values.")
        else:
            verified = False
            raise Warning("Tensor: Empty")

        # Coords
        if hasattr(input_tensor, 'coords') and input_tensor.coords is not None:
            if input_tensor.coords.casefold() == "cartesian":
                if not suppress_msgs:
                    print("Coords: " + input_tensor.coords)
            else:
                raise Warning("Non-cartesian coordinates are not supported at this time. Set .coords to 'cartesian'.")
        else:
            verified = False
            raise Warning("Coords: Empty")

        # Index
        if hasattr(input_tensor, 'index') and input_tensor.index is not None:
            if input_tensor.index.casefold() in ["contravariant", "covariant", "mixedupdown", "mixeddownup"]:
                if not suppress_msgs:
                    print("Index: " + input_tensor.index)
            else:
                verified = False
                raise Warning("Unknown index")
        else:
            verified = False
            raise Warning("Index: Empty")
    else:
        verified = False
        raise Warning("Tensor type field does not exits. Must be either Metric or Stress-Energy")
    return verified