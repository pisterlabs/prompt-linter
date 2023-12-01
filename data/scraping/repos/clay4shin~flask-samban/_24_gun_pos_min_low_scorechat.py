from flask import Blueprint, render_template, url_for, request, flash, current_app
from werkzeug.utils import redirect
from datetime import datetime
from google.cloud import language_v1
import openai
import os

from samban import db
from samban.models import Reply

from samban.forms_eng import IDForm, Reply1Form, Reply2Form

bp = Blueprint('cond24', __name__, url_prefix='/y2s1/cond24')

@bp.route('/', methods=('GET', 'POST'))
def index():
    current_app.logger.info("INFO 레벨로 출력")
    form = IDForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = Reply.query.filter_by(participant_id=form.ID.data).first()
        if user:
            flash("This is a duplicate participant ID. Please double-check your own participant ID and enter it again.")
        else:
            r = Reply(participant_id=form.ID.data, create_date=datetime.now(), condition=24)
            db.session.add(r)
            db.session.commit()
            return redirect(url_for('cond24.direct', participant_id=form.ID.data))
    return render_template('y2s1/index_eng.html', form=form)

@bp.route('/direct/<int:participant_id>/', methods=('GET', 'POST'))
def direct(participant_id):
    form = Reply1Form()
    if request.method == 'POST' and form.validate_on_submit():
        Reply.query.filter(Reply.participant_id == participant_id).\
            update({'reply1': form.Reply1.data})
        db.session.commit()
        return redirect(url_for('cond24.result', participant_id=participant_id))
    return render_template('y2s1/exposure_gun_neg_low.html', participant_id=participant_id, form=form)

@bp.route('/direct/<int:participant_id>/result/', methods=('GET', 'POST'))
def result(participant_id):
    q1 = Reply.query.filter(Reply.participant_id == participant_id)
    # Google Cloud Client Library 시작 (english specific)
    client = language_v1.LanguageServiceClient()
    for row in q1:
        document = language_v1.Document(content=row.reply1, type_=language_v1.Document.Type.PLAIN_TEXT, language='en')
        sentiment = client.analyze_sentiment(
            request={"document": document}
        ).document_sentiment
        # Send document to OpenAI for revision
        revised_doc = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Rewrite the following online comment to sound more positive, respectful, and civil while still maintaining the key argument as it is. When you revise, make your best guess even if you don't understand the context fully. Do not ask for more clarification and just give us a version of revision in the best way as you can. Do not say you can't assist with the request or ask for a longer sentence or a context. Do not directly ask for more repsect and positivity from the writer."},
                {"role": "user", "content": row.reply1}
            ]
        )
        # sentiment analysis score & chatCPT revision DB에 저장
        q1.update({'score': sentiment.score, 'chatGPT': revised_doc['choices'][0]['message']['content']})
        db.session.commit()
    # sentiment.score 반올림하기 (소숫점 두번째 자리까지)
    score_round = round(sentiment.score, 2)
    revised_comment = revised_doc['choices'][0]['message']['content']
    return render_template('y2s1/result_scorechat_eng.html', participant_id=participant_id, score_round=score_round, revised_comment=revised_comment, row=row, condition=24)


@bp.route('/direct/<int:participant_id>/result/revise', methods=('GET', 'POST'))
def revise(participant_id):
    form = Reply2Form()
    q = Reply.query.filter(Reply.participant_id == participant_id)
    rr = q[0].chatGPT
    if request.method == 'POST' and form.validate_on_submit():
        q.update({'reply2': form.Reply2.data})
        # Google Cloud Client Library 시작 (english specific)
        client = language_v1.LanguageServiceClient()
        for row in q:
            document = language_v1.Document(content=row.reply2, type_=language_v1.Document.Type.PLAIN_TEXT,
                                            language='en')
            sentiment = client.analyze_sentiment(
                request={"document": document}
            ).document_sentiment
            # sentiment analysis score DB에 저장
            q.update({'score2': sentiment.score})
            db.session.commit()
        return redirect(url_for('cond24.bye', participant_id=participant_id))
    return render_template('y2s1/revise_gun_neg_low.html', participant_id=participant_id, form=form, rr=rr, condition=24)

@bp.route('/direct/<int:participant_id>/result/bye', methods=('GET', 'POST'))
def bye(participant_id):
    return render_template('y2s1/bye_eng.html', participant_id=participant_id)

