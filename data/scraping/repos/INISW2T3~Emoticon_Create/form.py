from flask import Flask, render_template, redirect, url_for, session,request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, TextAreaField
from wtforms.validators import InputRequired, NumberRange, ValidationError
from chatgpt import openai
import os



class ConceptForm(FlaskForm):
    concept = StringField('개념', validators=[InputRequired()])

class PersonalizeStatus(FlaskForm):
    status = SelectField('개인화 상태', choices=[('yes', '예'), ('no', '아니오')])

class EmotionCount(FlaskForm):
    happiness = IntegerField('Happiness', validators=[NumberRange(min=0, max=32)])
    sadness = IntegerField('Sadness', validators=[NumberRange(min=0, max=32)])
    anger = IntegerField('Anger', validators=[NumberRange(min=0, max=32)])
    disgust = IntegerField('Disgust', validators=[NumberRange(min=0, max=32)])
    neutral = IntegerField('Neutral', validators=[NumberRange(min=0, max=32)])
    surprise = IntegerField('Surprise', validators=[NumberRange(min=0, max=32)])
    fear = IntegerField('Fear', validators=[NumberRange(min=0, max=32)])

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators):
            return False
        total = (
            self.happiness.data + self.sadness.data + self.anger.data +
            self.disgust.data + self.neutral.data + self.surprise.data +
            self.fear.data
        )
        if total != 32:
            raise ValidationError('감정 개수의 합은 반드시 32여야 합니다.')
        return True

class EmotionDescribe(FlaskForm):
    happiness_description = TextAreaField('기쁨 감정 설명', validators=[InputRequired()])
    sadness_description = TextAreaField('슬픔 감정 설명', validators=[InputRequired()])
    anger_description = TextAreaField('분노 감정 설명', validators=[InputRequired()])
    disgust_description = TextAreaField('혐오 감정 설명', validators=[InputRequired()])
    neutral_description = TextAreaField('중립 감정 설명', validators=[InputRequired()])
    surprise_description = TextAreaField('놀람 감정 설명', validators=[InputRequired()])
    fear_description = TextAreaField('두려움 감정 설명', validators=[InputRequired()])
# from flask import Flask, render_template, redirect, url_for
# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField, SelectField, IntegerField
# from wtforms.validators import DataRequired, NumberRange, Length

# class ConceptForm(FlaskForm):
#     concept = StringField('Concept', validators=[DataRequired()])



# class EmotionCount(FlaskForm):
#     happiness = IntegerField('Happiness', validators=[DataRequired()])
#     sadness = IntegerField('Sadness', validators=[DataRequired()])
#     anger = IntegerField('Anger', validators=[DataRequired()])
#     fear = IntegerField('Fear', validators=[DataRequired()])
#     surprise = IntegerField('Surprise', validators=[DataRequired()])
#     disgust = IntegerField('Disgust', validators=[DataRequired()])
#     neutral = IntegerField('Neutral', validators=[DataRequired()])

# class EmotionDescribe(FlaskForm):
#     describe = StringField('Describe', validators=[DataRequired()])

# class PersonalizeStatus(FlaskForm):
#     status= StringField('Status', validators=[DataRequired()])
# class CreateForm(FlaskForm):
#     field_names = ['Describe_1', 'Describe_2', 'Describe_3','Describe_4','Describe_5','Describe_6','Describe_7','Describe_8','Describe_9','Describe_10',
#                    'Describe_11', 'Describe_12', 'Describe_13','Describe_14','Describe_15','Describe_16','Describe_17','Describe_18','Describe_19','Describe_20',
#                    'Describe_21', 'Describe_22', 'Describe_23','Describe_24','Describe_25','Describe_26','Describe_27','Describe_28','Describe_29','Describe_30',
#                    'Describe_31','Describe_32']

#     concept = StringField('Concept', validators=[DataRequired()])

#     hap_cnt = IntegerField('Happiness', validators=[DataRequired()])
#     sad_cnt = IntegerField('Sadness', validators=[DataRequired()])
#     ang_cnt = IntegerField('Anger', validators=[DataRequired()])
#     fea_cnt = IntegerField('Fear', validators=[DataRequired()])
#     sur_cnt = IntegerField('Surprise', validators=[DataRequired()])
#     dis_cnt = IntegerField('Disgust', validators=[DataRequired()])
#     neu_cnt = IntegerField('Neutral', validators=[DataRequired()])

#     for field_name in field_names:
#         locals()[field_name] = StringField(field_name, validators=[DataRequired()])

#     submit = SubmitField('Submit')