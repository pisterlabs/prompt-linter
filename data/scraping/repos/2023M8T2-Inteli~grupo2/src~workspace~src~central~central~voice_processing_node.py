import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai

class VoiceProcessingNode(Node):
    def __init__(self):
        super().__init__('voice_processing_node')
        self.subscription = self.create_subscription(
            String, 'voice_command', self.voice_command_callback, 10)
        self.publisher = self.create_publisher(String, 'llm_command', 10)
        self.get_logger().info('Voice Processing Node has been started.')
        self.client = openai
        self.log_publisher = self.create_publisher(String, "log_register", 10)

    def voice_command_callback(self, msg):
        self.log_publisher.publish(String(data=f'Voice Processing Node ativado"'))

        voice_file_path = msg.data
        transcript = self.transcribe_voice(voice_file_path)
        if transcript:
            self.log_publisher.publish(String(data=f'Transcript do Voice Processing Node: "{transcript}"'))
            self.publisher.publish(String(data=transcript))

    def transcribe_voice(self, voice_file_path):
        try:
            # Carregue o arquivo de áudio para a memória
            with open(voice_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Transcreva o áudio para texto usando o Whisper
            model = "whisper-1"  # Especifique o modelo adequado do Whisper aqui
            transcript = self.client.audio.transcribe(audio_data, model=model)
            transcription_text = transcript['data']['text']
            self.get_logger().info(f'Transcription: {transcription_text}')
            return transcription_text
        except Exception as e:
            self.get_logger().error(f'Failed to transcribe voice file: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    voice_processing_node = VoiceProcessingNode()
    rclpy.spin(voice_processing_node)
    voice_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
