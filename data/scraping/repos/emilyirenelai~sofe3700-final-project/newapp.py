import mysql.connector

class Advice():
    def __init__(self, aid: int, eid: int, aname: str, link: str, alt: str):
        self.aid = aid
        self.eid = eid
        self.aname = aname
        self.link = link
        self.alt = alt

class Emotion():
    def __init__(self, emotionid: int, emotion: str, emoji: str):
        self.emotionid = emotionid
        self.emotion = emotion
        self.emoji = emoji


class Exercise():
    def __init__(self, exid: int, eid: int, exname: str, link: str, alt: str):
        self.exid = exid
        self.eid = eid
        self.exname = exname
        self.link = link
        self.alt = alt

class Journal():
    def __init__(self, jid: int, eid: int, jname: str, content: str):
        self.jid = jid
        self.eid = eid
        self.jname = jname
        self.content = content
        
    def add_emotion(self, emotion: str):
        self.emotion = emotion
        
    def add_emoji(self, emoji: str):
        self.emoji = emoji

class Music():
    def __init__(self, mid: int, eid: int, mname: str, link: str, alt: str):
        self.mid = mid
        self.eid = eid
        self.mname = mname
        self.link = link
        self.alt = alt

class User():
    def __init__(self, userid: int, Pass: str, fname: str, lname: str, email: str, dob: str, pronouns: str):
        self.userid = userid
        self.Pass = Pass
        self.fname = fname
        self.lname = lname
        self.email = email
        self.dob = dob
        self.pronouns = pronouns

class Database():
    def __init__(self):
        self.db = mysql.connector.connect(
            host="jeremymark.ca",
            user="jeremy",
            password="jeremy",
            port=3306,
            database="MindNBody"
        )

    # =============================
    # Advices functions
    # =============================

    def get_all_advices(self):
        query = "SELECT * FROM Advices;"
        cursor = self.db.cursor()
        cursor.execute(query)

        advices = []

        result = cursor.fetchall()

        for a in result:
            advice = Advice(int(a[0]), int(a[1]), a[2], a[3], a[4])
            advices.append(advice)

        return advices

    # Get the list of advices given a particular eid
    def get_advices(self, eid: int):

        query = "SELECT * FROM Advices WHERE eid = " + str(eid) + ";"
        cursor = self.db.cursor()
        cursor.execute(query)

        advices = []

        result = cursor.fetchall()

        for a in result:
            advice = Advice(int(a[0]), int(a[1]), a[2], a[3], a[4])
            advices.append(advice)

        return advices

    # =============================
    # Emotions functions
    # =============================

    def get_all_emotions(self):
        query = "SELECT * FROM Emotions;"

        cursor = self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()

        emotions = []

        for row in result:
            emotion = Emotion(row[0], row[1], row[2])
            emotions.append(emotion)
        
        return emotions

    def get_emotion_emotion(self, eid: int):
        emotions = self.get_all_emotions()
        for emotion in emotions:
            if emotion.emotionid == eid:
                return emotion.emotion
        return None
    
    def get_emotion_emoji(self, eid: int):
        emotions = self.get_all_emotions()
        for emotion in emotions:
            if emotion.emotionid == eid:
                return emotion.emoji
        return None

    # =============================
    # Exercises functions
    # =============================

    # Get the list of exercises given a particular eid
    def get_exercises(self, eid: int):
        query = "SELECT * FROM Exercises WHERE eid = " + str(eid) + ";"
        cursor = self.db.cursor()
        cursor.execute(query)

        exercises = []

        result = cursor.fetchall()

        for exer in result:
            exercise = Exercise(int(exer[0]), int(exer[1]), exer[2], exer[3], exer[4])
            exercises.append(exercise)

        return exercises
    
    def get_all_exercises(self):
        query = "SELECT * FROM Exercises;"
        cursor = self.db.cursor()
        cursor.execute(query)

        exercises = []

        result = cursor.fetchall()

        for exer in result:
            exercise = Exercise(int(exer[0]), int(exer[1]), exer[2], exer[3], exer[4])
            exercises.append(exercise)

        return exercises

    # =============================
    # Journal functions
    # =============================

    def insert_raw_journal(self, eid: int, jname: str, content: str) -> int:
        cursor = self.db.cursor()
        query = "INSERT INTO Journal (eid, jname, content) VALUES ("+ str(eid) + ", \"" + str(jname) + "\", \"" + str(content) + "\");"
        cursor.execute(query)

        self.db.commit()
        jid = cursor.lastrowid

        return jid

    def insert_journal(self, userid: int, eid: int, jname: str, content: str) -> int:
        jid = self.insert_raw_journal(eid, jname, content)
        
        query = "INSERT INTO JLibrary (jid, userid) VALUES (" + str(jid) + "," + str(userid) + ");"
        cursor = self.db.cursor()
        cursor.execute(query)
        self.db.commit()

        return jid
    
    def get_journal_by_id(self, jid: int) -> Journal:
        query = "SELECT * FROM Journal WHERE jid = " + str(jid) + ";"
        cursor = self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        if(len(result) == 1):
            jt = result[0]
            j = Journal(int(jt[0]), int(jt[1]), jt[2], jt[3])
            j.add_emotion(self.get_emotion_emotion(j.eid))
            j.add_emoji(self.get_emotion_emoji(j.eid))
            return j
        return None

    # Get a list of journal ids ([1, 2, 3,..]) given a userid (e.g. 3)
    def get_all_journal_ids(self, userid: int):
        query = "SELECT * FROM JLibrary WHERE userid=" + str(userid) + ";"
        cursor = self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()

        journal_ids = []

        for a in result:
            journal_ids.append(a[0])

        return journal_ids
    # =============================
    # Music functions
    # =============================
    
    def get_music(self, eid: int):
        query = "SELECT * FROM Music WHERE eid=" + str(eid) + ";"
        cursor = self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()

        musics = []

        for a in result:
            music = Music(int(a[0]), int(a[1]), a[2], a[3], a[4])
            musics.append(music)

        return musics

    def get_all_music(self):
        query = "SELECT * FROM Music;"
        cursor = self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        
        musics = []

        for a in result:
            music = Music(int(a[0]), int(a[1]), a[2], a[3], a[4])
            musics.append(music)
        return musics
    
    # =====
    # Users functions
    # =====

    def get_all_users(self):
        query = "SELECT * FROM Users;"
        cursor = self.db.cursor()
        cursor.execute(query)

        users = []

        result = cursor.fetchall()
        for x in result:
            user = User(int(x[0]), x[1], x[2], x[3], x[4], x[5], x[6])
            users.append(user)
        
        return users

    def get_user_by_id(self, userid: int):
        query = "SELECT * FROM Users WHERE userid=" + str(userid) + ";"
        cursor= self.db.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        if(len(result) == 1):
            x = result[0]
            return User(int(x[0]), x[1], x[2], x[3], x[4], x[5], x[6])
        return None
    
    def insert_user(self, Pass, fname, lname, email, dob, pronouns):
        query = "INSERT INTO Users (pass, fname, lname, email, dob, pronouns) VALUES (\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\");" %(Pass, fname, lname, email, dob, pronouns)

        cursor = self.db.cursor()
        cursor.execute(query)
        self.db.commit()
        
        return (cursor.lastrowid) # returns userid (auto incremented)

    def login_user(self, Pass, email):
        query = 'SELECT * FROM Users WHERE email = "%s" AND pass = "%s";' %(email, Pass)
        cursor = self.db.cursor()
        cursor.execute(query)
        
        result = cursor.fetchall()
        print(result)
        print(result)
        print(result)
        print(result)
        if(len(result) == 1):
            x = result[0]
            return User(int(x[0]), x[1], x[2], x[3], x[4], x[5], x[6])
        return None

    

if __name__ == '__main__':
    database = Database()

    # Example 1 : Getting emotions from database
    print("\n\nExample 1\n\n")
    emotions = database.get_all_emotions()
    for e in emotions:
        mystr = "(" + str(e.emotionid) + ", " + str(e.emotion) + ", " + str(e.emoji) + ")"
        print(mystr)

    # Example 2 : Inserting a happy journal into Journal table
    print("\n\nExample 2\n\n")
    # 2a. Get the emotionid for `hap`
    emotionid = -1 # we want to find the `happy` emotionid
    for e in emotions:
        if e.emotion == "hap":
            emotionid = e.emotionid

    # 2b. Insert it into database
    userid = 3
    if emotionid != -1:
        # we are guaranteed that `emotionid` is the happy emotionid
        jname = "I am so happy"
        content = "I had ice cream today"
        jid = database.insert_journal(userid, emotionid, jname, content)
        print("Inserted (%s, %s, %s, %s) into Journal" %(str(jid), str(emotionid), jname, content))
        print("Inserted (%s, %s) into JLibrary" %(str(jid), str(userid)))

    # Example 3 : Get advice from a particular eid
    print("\n\nExample 3\n\n")
    emotionid = 2
    advices = database.get_advices(emotionid)
    for advice in advices:
        print(str(advice.aid) + ": " + advice.aname + " | " + advice.link)

    # Example 4 : Get exercises from a particular eid
    print("\n\nExample 4\n\n")
    emotionid = 3
    exercises = database.get_exercises(emotionid)
    for exercise in exercises:
        print(str(exercise.exid) + ": \"" + str(database.get_emotion_emoji(emotionid)) + "\" means you should do exercise: " + str(exercise.exname) + ": (" + str(exercise.link) + ")")

    # Example 5 : Get all the journal ids from a userid
    print("\n\nExample 5\n\n")
    userid = 3
    journal_ids = database.get_all_journal_ids(userid)
    print(str(journal_ids))

    # Example 6: Get Journal by id == 1
    print("\n\nExample 6\n\n")
    jid = 1
    journal = database.get_journal_by_id(jid)
    print("Journal: (%s, %s, %s, %s)" %(str(journal.jid), str(journal.eid), journal.jname, journal.content))

    # Example 7 : Insert a new user
    print("\n\nExample 7\n\n")
    Pass = "123"
    fname = "Jem"
    lname = "Tom"
    email = "jem.tom@ontariotechu.net"
    dob = "2002/06/23"
    pronouns = "M" 
    userid = database.insert_user(Pass, fname, lname, email, dob, pronouns)
    print("Inserted (%s, %s, %s, %s, %s, %s, %s, %s)" %(str(userid), Pass, fname, lname, email, dob, pronouns, userid))

    # Example 8: Get user by id
    print("\n\nExample 8\n\n")
    userid = 1
    user = database.get_user_by_id(userid)
    print("Got User (%s, %s, %s, %s, %s, %s, %s)" %(str(user.userid), user.Pass, user.fname, user.lname, user.email, user.dob, user.pronouns))

    # Example 9: Get music by eid
    print("\n\nExample 9\n\n")
    eid = 2
    musics = database.get_music(eid)
    print("Got Musics: [")
    for music in musics:
        to_print = "Music (%s, %s, %s, %s, %s)" %(str(music.mid), str(music.eid), music.mname, music.link, music.alt)
        print("\t%s" %to_print)
    print("]\n")

    # Example 10 : Get all music
    print("\n\nExample 10\n\n")
    musics = database.get_all_music()
    print("Got Musics: [")
    for music in musics:
        to_print = "Music (%s, %s, %s, %s, %s)" %(str(music.mid), str(music.eid), music.mname, music.link, music.alt)
        print("\t%s" %to_print)
    print("]\n")

    # Example 11 : Get the music from the `hap` emotion
    print("\n\nExample 11\n\n")
    emotion = 'hap'
    emotionid = -1
    emotions = database.get_all_emotions()
    for e in emotions:
        if (e.emotion == emotion):
            emotionid = e.emotionid
            break
    if(emotionid != -1):
        musics = database.get_music(emotionid)
        for music in musics:
            print(str(music))

    # Example 12 : Put a journal using AI
    print("\n\nExample 12\n\n")
    userid = 1
    eid = -1 # to be evaluated by AI
    jname = "A normal day"
    content = "Today was a normal day"
    from cohere_mood_training import get_emotion
    emotion_name = get_emotion(content)
    for emotion in  database.get_all_emotions():
        if (emotion.emotion == emotion_name):
            eid = emotion.emotionid
            break
    if(eid != -1): # we found an eid
        database.insert_journal(userid, eid, jname, content)
        print("Inserted (%s, %s, %s, %s) into Journal" %(str(userid), str(eid), jname, content))

    
