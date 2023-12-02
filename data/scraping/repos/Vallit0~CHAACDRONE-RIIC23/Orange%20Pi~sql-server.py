import pyodbc
import random
import datetime
#import openai

# Configura tu clave de API de OpenAI
#api_key = "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"

# Función para generar descripciones usando la API de OpenAI
def generate_description(humidity, temperature, pressure, soil_temperature, soil_humidity):
    #prompt = "Genera una descripción breve sobre las variables que te dare mas adelante y que me dicen de mi campo, variables de Atterberg y estado de la tierra: de los siguientes valores humedad" + str(humidity) + " humedad en tierra" + str(soil_humidity) + ": presion" + str(pressure) + ": temperatura"+ str(temperature) + " temperatura en la tierra" + str(soil_temperature) + "alta verapaz, guatremala"
    #response = openai.Completion.create(
    #    engine="text-davinci-002",
    #    prompt=prompt,
    #    max_tokens=50,  # Ajusta la longitud de la respuesta según sea necesario
    #    api_key=api_key
    #)
    return "descripcion" #response.choices[0].text.strip() "

def main():
    # Configura la cadena de conexión
    server = "chaacserver.database.windows.net"
    database = "chaacdb"
    username = "chaac"
    password = "riic4.02023"
    driver = "{ODBC Driver 17 for SQL Server}"  # Asegúrate de usar el controlador correcto

    connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"

    # Intenta establecer una conexión
    try:
        conn = pyodbc.connect(connection_string)
        print("Conexión exitosa a la base de datos")
    except pyodbc.Error as e:
        print("Error al conectar a la base de datos:", str(e))
        exit()

    try:
        cursor = conn.cursor()

        # Nombre de la tabla en la que deseas insertar datos
        tabla = "chaacTable"

        # ------- INSERTAMOS LOS DATOS --------- FINALMENTE
        # Datos a insertar en la tabla
        # Consulta SQL de inserción

        # Genera valores aleatorios
        humidity = random.uniform(30, 60)
        temperature = random.uniform(20, 30)
        pressure = random.uniform(900, 1100)  # Replace with appropriate pressure range
        soil_temperature = random.uniform(10, 30)  # Replace with appropriate soil_temp range
        soil_humidity = random.uniform(40, 80)  # Replace with appropriate soil_hum range
        UTC = "UTC+0"  # Huso horario UTC

        # Genera una descripción usando la API de OpenAI
        description = generate_description(humidity, temperature, pressure, soil_temperature, soil_humidity)

        danger = random.choice(["Bajo", "Moderado", "Alto"])  # Nivel de peligro aleatorio

        # Consulta SQL de inserción
        insert_query = f"INSERT INTO chaacTable (temperature, humidity, pressure, description, soil_humidity, soild_temperature) VALUES (?, ?, ?, ?, ?, ?)"

        # Ejecuta la consulta de inserción con los valores generados aleatoriamente
        cursor.execute(insert_query,
                       (temperature, humidity, pressure, description, soil_humidity, soil_temperature))

        # Confirma la transacción
        conn.commit()

        print("Datos insertados correctamente en la tabla", tabla)

    except pyodbc.Error as e:
        # Si ocurre un error, puedes revertir la transacción
        conn.rollback()
        print("Error al insertar datos:", str(e))
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    for i in range(0, 200):
        main()
