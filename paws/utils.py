from .settings import (
    SIGMOID_ACTIVATION,
    MASS_SCALE,
    MASS_RANGE,
    SEMI_WEAKLY_PARAMETERS
)

def get_parameter_transform(parameter: str):
    from aliad.interface.keras import activations

    if parameter in ['m1', 'm2']:
        return activations.Scale(1 / MASS_SCALE)
    elif parameter in ['mu', 'alpha']:
        if SIGMOID_ACTIVATION:
            return actiations.Sigmoid()
        if parameter == 'mu':
            return activations.Exponential()
        return activations.Linear()
    raise ValueError(f'unknown parameter for semi-weakly model: {parameter}')

def get_parameter_inverse_transform(parameter: str):
    return get_parameter_transform(parameter).inverse

def get_parameter_transforms():
    transforms = {}
    for parameter in SEMI_WEAKLY_PARAMETERS:
        transforms[parameter] = get_parameter_transform(parameter)
    return transforms

def get_parameter_inverse_transforms():
    inverse_transforms = {}
    for parameter in SEMI_WEAKLY_PARAMETERS:
        inverse_transforms[parameter] = get_parameter_inverse_transform(parameter)
    return inverse_transforms

def get_parameter_regularizer(parameter: str):
    from aliad.interface.tensorflow.regularizers import MinMaxRegularizer
    
    if parameter in ['m1', 'm2']:
        mass_range = (MASS_RANGE[0] * MASS_SCALE, MASS_RANGE[1] * MASS_SCALE)
        return MinMaxRegularizer(*mass_range)
    elif parameter == 'mu':
        if SIGMOID_ACTIVATION:
            return None
        return MinMaxRegularizer(-10.0, 0.0)
    elif parameter == 'alpha':
        if SIGMOID_ACTIVATION:
            return None
        return MinMaxRegularizer(0.0, 1.0, 10.0)
    raise ValueError(f'unknown parameter for semi-weakly model: {parameter}')