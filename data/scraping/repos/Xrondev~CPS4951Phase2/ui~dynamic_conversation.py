"""
Dynamically display the conversation between the user and the chat model from OpenAI
"""
import os

from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSpacerItem, \
    QVBoxLayout, QWidget
from PySide6.QtWidgets import QMessageBox

from utils.chat_session_maintainer import ChatSessionMaintainer

dirname = os.path.dirname(__file__)


class ConversationWidget(QWidget):
    """
    This widget is used to display the conversation between the user and the GPT-3 model.
    It is on the right side of the main window.
    """

    def __init__(self):
        super().__init__()

        # Initialize UI elements
        self.layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

        # Create sender icons
        self.sender_icons = {
            'gpt': QIcon(os.path.join(dirname, 'icon/robot.png')),
            'user': QIcon(os.path.join(dirname, 'icon/account-edit.png'))
        }

        # Create chat session maintainer
        self.csm = ChatSessionMaintainer()

    def add_message(self, sender, message) -> None:
        """
        Add a message to the conversation widget
        :param sender: 'gpt' or 'user'
        :param message: The message to be displayed, if the message is empty,
        a popup dialog box will be created
        :return: None
        """

        if message == '':
            # create a popup dialog box to tell the user that the message is empty
            QMessageBox.critical(self, "Error", "Message is empty")
            return

        # Create message label
        message_label = QLabel(message)
        message_label.setWordWrap(True)

        # Create icon label
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        icon_label.setMaximumWidth(32)
        icon_label.setMaximumHeight(32)
        if sender == 'gpt':
            icon_label.setPixmap(self.sender_icons['gpt'].pixmap(QSize(32, 32)))
            message_layout = QHBoxLayout()
            message_layout.addWidget(icon_label)
            message_layout.addWidget(message_label)
        elif sender == 'user':
            icon_label.setPixmap(self.sender_icons['user'].pixmap(QSize(32, 32)))
            message_layout = QHBoxLayout()
            message_layout.addWidget(message_label)
            message_layout.addWidget(icon_label)
        else:
            # create a popup dialog box to tell the user that the sender is invalid
            QMessageBox.critical(self, "Error", "Invalid sender")
            raise ValueError('Invalid sender')

        # Create spacer widget
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Add message layout to scroll layout
        message_layout.addSpacerItem(spacer)
        message_layout.setContentsMargins(0, 0, 0, 0)

        # Create widget container for message and icon
        message_widget = QWidget()
        message_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        message_widget.setLayout(message_layout)
        if sender == 'gpt':
            self.scroll_layout.addWidget(message_widget)
        elif sender == 'user':
            self.scroll_layout.addWidget(message_widget, 0, Qt.AlignmentFlag.AlignRight)

        # Add message content to the chat session maintainer
        # self.csm.add_message(message)

        # Scroll to bottom
        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()))

    def clear_conversation(self) -> None:
        """
        clear the display of the conversation. This will also clear the chat session maintainer.
        :return: None
        """
        print('clearing conversation')
        # Remove all messages from the scroll layout
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)

        self.csm.clear()
