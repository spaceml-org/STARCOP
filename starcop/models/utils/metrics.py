from torchmetrics.functional import mean_squared_error

def rmse(input, target):
    return mean_squared_error(input, target, squared=False) # < squared: returns RMSE value if set to False
