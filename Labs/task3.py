import numpy as np


y_real = [100, 150, 200, 250, 300]  # Valores reales
y_pred = [110, 140, 210, 240, 500]


def mae_calc(dt_real, dt_pred):

    if len(dt_real) != len(dt_pred):
        raise ValueError('`dt_real` y `dt_pred` deben tener la misma longitud')
    n = len(dt_real)
    if n == 0:
        raise ValueError('Las listas de entrada no deben estar vacías')
    # Suma de errores absolutos dividida por n
    mae = sum(abs(dt_real[i] - dt_pred[i]) for i in range(n)) / n
    return mae

def rmse_calc(dt_real, dt_pred):

    if len(dt_real) != len(dt_pred):
        raise ValueError('`dt_real` y `dt_pred` deben tener la misma longitud')
    n = len(dt_real)
    if n == 0:
        raise ValueError('Las listas de entrada no deben estar vacías')
    # Raíz cuadrada de la media de los errores al cuadrado
    rmse = (sum((dt_real[i] - dt_pred[i]) ** 2 for i in range(n)) / n) ** 0.5
    return rmse


if __name__ == '__main__':
    mae = mae_calc(y_real, y_pred)
    rmse = rmse_calc(y_real, y_pred)

    print(f"Mean Absolute Error (MAE):\n {mae}")
    print(f"Root Mean Squared Error (RMSE):\n {rmse}")
    print("El RMSE penaliza más los errores grandes debido a la elevación al cuadrado de las diferencias, mientras que el MAE trata todos los errores de manera uniforme.")
    print("Bajo el ejemplo de dosis médicas, un error grande en la predicción podría tener consecuencias graves, por lo que el RMSE sería una métrica más adecuada para evaluar el rendimiento del modelo.")