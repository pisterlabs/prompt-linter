import os
import re
import cohere
from flask import (
    Flask, flash, render_template,
    redirect, request, session, url_for, abort)
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
if os.path.exists("env.py"):
    import env


app = Flask(__name__)

# ------------------------------------------------------------- CONFIG  #
app.config["MONGO_DBNAME"] = os.environ.get("MONGO_DBNAME")
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")

mongo = PyMongo(app)


# ------------------------------------------------------------- HOMEPAGE  #
@app.route("/")
def index():
    return render_template('index.html', page="index")


# ------------------------------------------------------------- USERS #
@app.route("/register", methods=["GET", "POST"])
# Render register page
def register():
    if request.method == "POST":
        # check if username already exists in db
        existing_user = mongo.db.users.find_one(
            {"username": request.form.get("username").lower()})

        if existing_user:
            flash("Username Already Exists")
            return redirect(url_for("register"))

        register = {
            "username": request.form.get("username").lower(),
            "email": request.form.get("email").lower(),
            "password": generate_password_hash(request.form.get("password"))
        }
        mongo.db.users.insert_one(register)

        # put the new user into session cookie
        session["user"] = request.form.get("username").lower()
        flash("Registration Successful!")
        return redirect(url_for("get_reviews", username=session["user"]))

    return render_template("register.html", page="register")


@app.route("/login", methods=["GET", "POST"])
# Render login page
def login():
    if request.method == "POST":
        # check if username exists in database
        existing_user = mongo.db.users.find_one(
            {"username": request.form.get("username").lower()})

        if existing_user:
            # ensure hashed password matches user input
            if check_password_hash(
                    existing_user["password"],
                    request.form.get("password")):
                session["user"] = request.form.get("username").lower()
                flash("Hi, {}".format(
                    request.form.get("username")))
                return redirect(url_for(
                    "get_reviews", username=session["user"]))
            else:
                # invalid password match
                flash("Incorrect User Details")
                return redirect(url_for("login"))

        else:
            # username doesn't exist
            flash("Incorrect User Details")
            return redirect(url_for("login"))

    return render_template("login.html", page="login")


@app.route("/logout")
# Logout Functionality
def logout():
    # remove user from session cookies
    flash("You've been logged out")
    session.pop("user")
    return redirect(url_for("login"))


# ------------------------------------------------------------- REVIEWS  #
def shorten_text(text, num_sentences):
  """Shortens the given text and makes sure it ends after a full stop.

  Args:
    text: The text to shorten.
    num_sentences: The number of sentences to return.

  Returns:
    A shorter response text containing the first `num_sentences` sentences from the given text.
  """

  # Split the text into a list of sentences.
  sentences = re.split(r"[.!?]\s+", text)

  # Iterate over the list of sentences and add the first `num_sentences` sentences to a new list.
  shorter_sentences = []
  for i in range(num_sentences):
    if i < len(sentences):
      shorter_sentences.append(sentences[i])
    else:
      break

  # Join the sentences in the new list with a full stop at the end of each sentence.
  shorter_text = ". ".join(shorter_sentences)

  # Return the shorter response text.
  return shorter_text


@app.route("/get_reviews")
# Render reviews
def get_reviews():
    reviews = mongo.db.reviews.find()
    co = cohere.Client(os.environ.get("BEARER_TOKEN"))

    response = co.generate(
    prompt='Why is the sky blue?',
    max_tokens=200,
    )

    # Print the response text before calling the `shorter_text` function on it.
    print(response.generations[0].text)

    # Access the generated text
    response_text = response.generations[0].text

    # Shorten the response text to 3 sentences.
    shorter_text = shorten_text(response_text, 3)

    # Add a full stop to the end of the shorter response text, if it does not already have one.
    if shorter_text[-1] != ".":
      shorter_text += "."

    print(shorter_text)
    return render_template("reviews.html", reviews=reviews, page="get_reviews", response_text=shorter_text)




@app.route("/search", methods=["GET", "POST"])
# Search functionality
def search():
    query = request.form.get("query")
    reviews = list(mongo.db.reviews.find({"$text": {"$search": query}}))

    # If no review found show flash msg
    if len(reviews) == 0:
        flash("No review found matching your search criteria!")
        return redirect(url_for("get_reviews"))

    return render_template("reviews.html", reviews=reviews, page="get_reviews")


@app.route("/write_review", methods=["GET", "POST"])
# Render write review page
def write_review():
    # Only users can write reviews
    if not session.get("user"):
        return render_template("403.html")

    # Add review to db
    if request.method == "POST":
        review = {
            "bar_name": request.form.get("bar_name"),
            "review": request.form.get("review"),
            "fav_drink": request.form.get("fav_drink"),
            "location": request.form.get("location"),
            "created_by": session["user"]
        }
        mongo.db.reviews.insert_one(review)
        flash("Review Successfully Added!")
        return redirect(url_for("get_reviews"))

    bars = mongo.db.bars.find().sort("bar_name", 1)
    return render_template("write_review.html", bars=bars, page="write_review")


@app.route("/edit_review/<review_id>", methods=["GET", "POST"])
# Render edit review page
def edit_review(review_id):
    # Only users can edit reviews
    if not session.get("user"):
        return render_template("403.html")

    # Edit review on db
    if request.method == "POST":
        submit = {
            "bar_name": request.form.get("bar_name"),
            "review": request.form.get("review"),
            "fav_drink": request.form.get("fav_drink"),
            "location": request.form.get("location"),
            "created_by": session["user"]
        }
        mongo.db.reviews.update({"_id": ObjectId(review_id)}, submit)
        flash("Review Successfully Updated!")
        return redirect(url_for("get_reviews"))

    review = mongo.db.reviews.find_one({"_id": ObjectId(review_id)})
    bars = mongo.db.bars.find().sort("bar_name", 1)
    return render_template(
        "edit_review.html", review=review, bars=bars, page="edit_review")


@app.route("/delete_review/<review_id>")
# Delete functionality
def delete_review(review_id):
    # Only users can delete reviews
    if not session.get("user"):
        return render_template("403.html")

    mongo.db.reviews.remove({"_id": ObjectId(review_id)})
    flash("Review Successfully Deleted!")
    return redirect(url_for("get_reviews"))


# ---------------------------------------------------------- ERROR HANDLERS #
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403


# ---------------------------------------------------------- RUN THE APP  #
if __name__ == "__main__":
    app.run(host=os.environ.get("IP"),
            port=int(os.environ.get("PORT")),
            debug=False)
