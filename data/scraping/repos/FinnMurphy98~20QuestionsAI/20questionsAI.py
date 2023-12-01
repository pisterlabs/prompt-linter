from app import create_app, db, socketio
from app.models import User, Game, Message
import openai

app = create_app()
openai.api_key = app.config.get('OPENAI_KEY')

if __name__ == '__main__':
    socketio.run(app, debug=True)

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Game': Game, 'Message': Message}