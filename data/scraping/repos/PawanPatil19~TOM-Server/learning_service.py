from enum import Enum

import math
import statistics
import modules.utilities.time as time_utility
import modules.utilities.image_utility as image_utility
from modules.dataformat.data_types import DataTypes
from . import learning_data_handler as learning_data_handler
from . import learning_display as learning_display

from modules.langchain_llm.LangChainTextGenerator import LangChainTextGenerator as TextGenerator
from modules.cloud_vision.VisionClient import VisionClient as TextDetector


class LearningConfig:
    finger_pointing_trigger_duration_seconds = 2
    finger_pointing_location_offset_percentage = 0.2  # 20% of the screen width/height
    finger_detection_duration_seconds = 1  # seconds (depends on the client)
    finger_pointing_buffer_size = math.ceil(
        finger_pointing_trigger_duration_seconds / finger_detection_duration_seconds)
    learning_data_display_duration_millis = 5000
    text_data_read_duration_millis = 3000
    finger_data_checking_duration_seconds = 0.2


_text_generator = None
_learning_map = {}

_image_detector = None
_finger_pose_buffer = []

_learning_data_sent_time = 0
_learning_data_has_sent = False

_text_detection_sent_time = 0
_text_detector = None


def _get_text_generator():
    global _text_generator

    if _text_generator is None:
        _text_generator = TextGenerator(0)

    return _text_generator


def _get_text_detector():
    global _text_detector

    if _text_detector is None:
        _text_detector = TextDetector()

    return _text_detector


def get_learning_data(object_of_interest, text_content, speech_content):
    global _learning_map

    if object_of_interest is None:
        previous_content = None
        prompt = ""
    else:
        previous_content = _learning_map.get(object_of_interest)
        prompt = f"A *{object_of_interest}* is here. "

    if previous_content is not None and text_content is None:
        return previous_content

    if text_content is not None:
        prompt += f"There are some texts, '{text_content}'. "

    if speech_content is not None:
        prompt += f"{speech_content}. Provide only the answer in one sentence."
    else:
        prompt += "Briefly describe about it in one sentence. Provide only the answer."

    print(f"Prompt: {prompt}")

    text_generator = _get_text_generator()
    learning_content = ""
    try:
        learning_content = text_generator.generate_response(prompt)
    except Exception as e:
        print(f"OpenAPI error: {e}")

    if object_of_interest is not None:
        _learning_map[object_of_interest] = learning_content

    return learning_content


def _handle_learning_requests(finger_pointing_data=None):
    # handle finger pointing data
    if finger_pointing_data is not None:
        _handle_finger_pose(finger_pointing_data)

    # handle learning requests from socket
    socket_data = learning_data_handler.get_socket_data()
    # skip if no data
    if socket_data is None:
        time_utility.sleep_seconds(LearningConfig.finger_data_checking_duration_seconds)
        return

    # decode data
    socket_data_type, decoded_data = learning_data_handler.get_decoded_socket_data(socket_data)
    if socket_data_type is None:
        return

    if socket_data_type == DataTypes.FINGER_POINTING_DATA:
        _handle_finger_pose(decoded_data)
    elif socket_data_type == DataTypes.REQUEST_LEARNING_DATA:
        print('Received learning request')
        # FIXME: ideally send data based on the request
    else:
        print(f'Unsupported data type: {socket_data_type}')


def _has_speech_data(finger_pose_data):
    # FIXME: this is not a good way to send speech data
    return finger_pose_data.details != ""


def _handle_finger_pose(finger_pose_data):
    global _finger_pose_buffer

    # see whether the pose if pointing to an object for a certain duration based on history
    if len(_finger_pose_buffer) >= LearningConfig.finger_pointing_buffer_size:
        prev_avg_camera_x = statistics.mean([_data.camera_x for _data in _finger_pose_buffer])
        prev_avg_camera_y = statistics.mean([_data.camera_y for _data in _finger_pose_buffer])

        print(f'Current:: x: {finger_pose_data.camera_x}, y: {finger_pose_data.camera_y}, '
              f'Previous (avg):: x: {prev_avg_camera_x}, y: {prev_avg_camera_y}')

        # remove the oldest data
        _finger_pose_buffer.pop(0)

        # if same location, identify object data and send it to the client
        if same_pointing_location(prev_avg_camera_x, prev_avg_camera_y, finger_pose_data,
                                  LearningConfig.finger_pointing_location_offset_percentage) or _has_speech_data(
            finger_pose_data):
            object_of_interest, text_content, speech_content = _get_detected_object_and_text_and_speech(
                finger_pose_data)
            if object_of_interest is not None or text_content is not None or speech_content is not None:
                _send_learning_data(object_of_interest, text_content, speech_content)

    # temporary store finger pose data
    _finger_pose_buffer.append(finger_pose_data)


def _send_learning_data(object_of_interest, text_content, speech_content):
    global _learning_data_sent_time, _learning_data_has_sent
    _learning_data_sent_time = time_utility.get_current_millis()
    _learning_data_has_sent = True

    learning_content = get_learning_data(object_of_interest, text_content, speech_content)
    formatted_content = learning_display.get_formatted_learning_details(learning_content)
    learning_data_handler.send_learning_data(object_of_interest, formatted_content)


def _clear_learning_data():
    global _learning_data_sent_time, _learning_data_has_sent

    if not _learning_data_has_sent or time_utility.get_current_millis() - _learning_data_sent_time < LearningConfig.learning_data_display_duration_millis:
        # do not clear
        return

    _learning_data_sent_time = time_utility.get_current_millis()
    _learning_data_has_sent = False

    learning_data_handler.send_learning_data()


def same_pointing_location(camera_x, camera_y, finger_pose_data, offset):
    finger_x = finger_pose_data.camera_x
    finger_y = finger_pose_data.camera_y

    return abs(camera_x - finger_x) < offset and abs(camera_y - finger_y) < offset


def _get_interest_objects_bounding_box(actual_bounding_box, image_width, image_height):
    # x1 = _get_value(min(int(actual_bounding_box[0]), finger_pointing_region[0]), image_width)
    # y1 = _get_value(min(int(actual_bounding_box[1]), finger_pointing_region[1]), image_height)
    # x2 = _get_value(max(int(actual_bounding_box[2]), finger_pointing_region[2]), image_width)
    # y2 = _get_value(max(int(actual_bounding_box[3]), finger_pointing_region[3]), image_height)

    x1 = _get_value(int(actual_bounding_box[0] + 0.5), image_width)
    y1 = _get_value(int(actual_bounding_box[1] + 0.5), image_height)
    x2 = _get_value(int(actual_bounding_box[2] + 0.5), image_width)
    y2 = _get_value(int(actual_bounding_box[3] + 0.5), image_height)

    return [x1, y1, x2, y2]


def _get_detected_object_and_text_and_speech(finger_pose_data):
    global _image_detector, _text_detection_sent_time

    if _image_detector is None:
        return None

    # get the speech content
    speech_content = None
    if finger_pose_data.details != "":
        speech_content = finger_pose_data.details

    # get the object of interest
    _, frame_width, frame_height, _ = _image_detector.get_source_params()
    finger_pointing_region = _get_image_region_from_camera(finger_pose_data.camera_x,
                                                           finger_pose_data.camera_y,
                                                           frame_width, frame_height,
                                                           LearningConfig.finger_pointing_location_offset_percentage)

    print(f'finger_pointing_region: {finger_pointing_region} - {speech_content}')

    # FIXME: clean up the code

    # get the object of interest
    frame = _image_detector.get_last_frame()
    detections = _image_detector.get_last_detection()
    # FIXME: check detection time and not null
    CLASS_ID_PERSON = 0
    detections = _image_detector.get_detection_in_region(detections, finger_pointing_region,
                                                         [CLASS_ID_PERSON])
    class_labels = _image_detector.get_class_labels()
    objects = [class_labels[class_id] for _, _, _, class_id, _ in detections]
    unique_objects = [s for s in set(objects)]
    print(f'Objects[{len(unique_objects)}]: {unique_objects}')

    object_of_interest = None
    interested_text_region = finger_pointing_region
    if len(unique_objects) > 0:
        object_of_interest = unique_objects[0]
        object_of_interest_class_id = list(class_labels.keys())[
            list(class_labels.values()).index(object_of_interest)]
        object_of_interest_bounding_box = \
            [bounding_box for bounding_box, _, _, class_id, _ in detections if
             class_id == object_of_interest_class_id][0]
        interested_text_region = _get_interest_objects_bounding_box(object_of_interest_bounding_box,
                                                                    frame_width, frame_height)
        # print(
        #     f'Object of interest: {object_of_interest}-{object_of_interest_class_id}-{object_of_interest_bounding_box}')

    # get the text content
    text_content = _detect_text(frame, interested_text_region)

    return object_of_interest, text_content, speech_content


# return [x1, y1, x2, y2]
def _get_image_region_from_camera(camera_x, camera_y, image_width, image_height, offset_length,
                                  camera_calibration=None):
    relative_x = camera_x
    relative_y = camera_y

    # if camera_calibration is None:  # FIXME: use camera calibration (instead of hard coded values for HL2)
    #     relative_x = (relative_x + 0.18) / (0.20 + 0.18)
    #     relative_y = (relative_y - 0.1) / (-0.12 - 0.1)

    image_x = int(relative_x * image_width)
    image_y = int(relative_y * image_height)
    image_offset = int(offset_length * image_width / 2)
    image_region = [_get_value(image_x - image_offset, image_width),
                    _get_value(image_y - image_offset, image_height),
                    _get_value(image_x + image_offset, image_width),
                    _get_value(image_y + image_offset, image_height)]
    return image_region


def _get_value(actual, max_value):
    if actual < 0:
        return 0
    if actual > max_value:
        return max_value
    return actual


def _detect_text(frame, text_region):
    global _text_detection_sent_time

    if time_utility.get_current_millis() - _text_detection_sent_time < LearningConfig.text_data_read_duration_millis:
        # do not detect text
        return None

    # height, width, channels = frame.shape
    # print(f'The dimensions of the frame are: Width = {width}, Height = {height}, Channels = {channels}')

    print(f'text_region: {text_region}')
    try:
        cropped_frame = image_utility.get_cropped_frame(frame, text_region[0], text_region[1],
                                                        text_region[2], text_region[3])
        image_png_bytes = image_utility.get_png_image_bytes(cropped_frame)
        image_utility.save_image_bytes('temp.png', image_png_bytes)

        text_contents, _, _ = _get_text_detector().detect_text_image_bytes(image_png_bytes)
        print(f'Texts[{len(text_contents)}]: {text_contents}')

        _text_detection_sent_time = time_utility.get_current_millis()

        if len(text_contents) > 0:
            return text_contents[0]
        return None

    except Exception as e:
        print("Error in detecting text: " + str(e))
        return None


def configure_learning(image_detector):
    global _image_detector

    _image_detector = image_detector


def update_learning(finger_pointing_data=None):
    _handle_learning_requests(finger_pointing_data)
    _clear_learning_data()
