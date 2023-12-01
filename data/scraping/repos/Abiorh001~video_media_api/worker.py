# import pika
# import os
# import requests
# import openai


# openai.api_key = "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"

  
# def process_audio_file_path(ch, method, properties, body):
#     audio_file_path = body.decode('utf-8')

#     # Read the audio file
#     audio_file = open(audio_file_path, "rb")
#     transcript = openai.Audio.translate("whisper-1", audio_file)
#     print(transcript)


    

#     # Acknowledge the message
#     ch.basic_ack(delivery_tag=method.delivery_tag)

# def consume_audio_paths_from_rabbitmq():
#     # Set up RabbitMQ connection
#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     channel = connection.channel()

#     # Declare the queue for incoming audio file paths
#     channel.queue_declare(queue='audio_file_paths')

#     # Set up the message consumer
#     channel.basic_consume(queue='audio_file_paths', on_message_callback=process_audio_file_path)

#     print("Waiting for audio file paths. To exit, press CTRL+C")
#     channel.start_consuming()

# if __name__ == "__main__":
#     consume_audio_paths_from_rabbitmq()

import subprocess

# Install pika library
subprocess.run(['pip', 'install', 'pika'], check=True)

# Install openai library
subprocess.run(['pip', 'install', 'openai'], check=True)

# Import installed libraries
import pika
import openai
import os


# RabbitMQ parameters
rabbitmq_user = 'your_username'
rabbitmq_password = 'your_password'
rabbitmq_vhost = 'your_vhost_name'
queue_name = 'audio_file_paths'

# OpenAI API key
openai.api_key = "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"

# Install RabbitMQ (replace with the appropriate package manager command)
try:
    install_cmd = 'sudo apt-get install -y rabbitmq-server'  # Adjust for your package manager if needed
    subprocess.run(install_cmd, shell=True, check=True, text=True)

    # Start RabbitMQ (replace with the appropriate command)
    start_cmd = 'sudo systemctl start rabbitmq-server'  # Adjust for your system
    subprocess.run(start_cmd, shell=True, check=True, text=True)

    # Create a virtual host
    create_vhost_cmd = f'sudo rabbitmqctl add_vhost {rabbitmq_vhost}'
    subprocess.run(create_vhost_cmd, shell=True, check=True, text=True)

    # Add RabbitMQ user and set password, associating it with the virtual host
    add_user_cmd = f'sudo rabbitmqctl add_user {rabbitmq_user} {rabbitmq_password}'
    subprocess.run(add_user_cmd, shell=True, check=True, text=True)

    set_user_tags_cmd = f'sudo rabbitmqctl set_user_tags {rabbitmq_user} administrator'
    subprocess.run(set_user_tags_cmd, shell=True, check=True, text=True)

    set_permissions_cmd = f'sudo rabbitmqctl set_permissions -p {rabbitmq_vhost} {rabbitmq_user} ".*" ".*" ".*"'
    subprocess.run(set_permissions_cmd, shell=True, check=True, text=True)

    print("RabbitMQ installed, virtual host created, and user configured successfully.")

except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

def process_audio_file_path(ch, method, properties, body):
    audio_file_path = body.decode('utf-8')

    # Read the audio file
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.translate("whisper-1", audio_file)
        print(transcript)

    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_audio_paths_from_rabbitmq():
    # Set up RabbitMQ connection
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', credentials=pika.PlainCredentials(rabbitmq_user, rabbitmq_password), virtual_host=rabbitmq_vhost))
    channel = connection.channel()

    # Declare the queue for incoming audio file paths
    channel.queue_declare(queue=queue_name)

    # Set up the message consumer
    channel.basic_consume(queue=queue_name, on_message_callback=process_audio_file_path)

    print("Waiting for audio file paths. To exit, press CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    consume_audio_paths_from_rabbitmq()
