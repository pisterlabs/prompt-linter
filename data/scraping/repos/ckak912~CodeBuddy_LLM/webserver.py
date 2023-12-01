import contextvars
import logging
import os
import sys
import traceback
from datetime import datetime
from distutils.log import debug

import tornado.ioloop
import ui_methods
from concurrent_log_handler import ConcurrentRotatingFileHandler
from content import *
from handlers import *
from helper import *
from tornado.auth import GoogleOAuth2Mixin
from tornado.web import *


class BaseHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "https://api.openai.com") # Only allow from openAI domain
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

def make_app(settings_dict):
    app = Application(
        [
            url(r"/", HomeHandler),
            url(r"/add_admin/([^/]+)", AddAdminHandler, name="add_admin"),
            url(r"/add_assistant/([^/]+)/([^/]+)", AddAssistantHandler, name="add_assistant"),
            url(r"/add_instructor/([^/]+)/([^/]+)", AddInstructorHandler, name="add_instructor"),
            url(r"/assignment/([^/]+)/([^/]+)", AssignmentHandler, name="assignment"),
            url(r"/available", AvailableCoursesHandler, name="available"),
            url(r"/caslogin", CASLoginHandler, name="caslogin"),
            url(r"/create_video_exercise/([^/]+)/([^/]+)", CreateVideoExerciseHandler, name="create_video_exercise"),
            url(r"/copy_assignment/([^/]+)/([^/]+)?", CopyAssignmentHandler, name="copy_assignment"),
            url(r"/copy_course/([^/]+)", CopyCourseHandler, name="copy_course"),
            url(r"/copy_exercise/([^/]+)/([^/]+)/([^/]+)?", CopyExerciseHandler, name="copy_exercise"),
            url(r"/course/([^/]+)", CourseHandler, name="course"),
            url(r"/courses", CoursesHandler, name="courses"),
            url(r"/delete_assignment/([^/]+)/([^/]+)?", DeleteAssignmentHandler, name="delete_assignment"),
            url(r"/delete_assignment_submissions/([^/]+)/([^/]+)?", DeleteAssignmentSubmissionsHandler, name="delete_assignment_submissions"),
            url(r"/delete_course/([^/]+)?", DeleteCourseHandler, name="delete_course"),
            url(r"/delete_course_submissions/([^/]+)?", DeleteCourseSubmissionsHandler, name="delete_course_submissions"),
            url(r"/delete_exercise/([^/]+)/([^/]+)/([^/]+)?", DeleteExerciseHandler, name="delete_exercise"),
            url(r"/delete_exercise_submissions/([^/]+)/([^/]+)/([^/]+)?", DeleteExerciseSubmissionsHandler, name="delete_exercise_submissions"),
            url(r"/delete_request/([^/]+)/([^/]+)/([^/]+)/([^/]+)", DeleteHelpRequestHandler, name="delete_request"),
            url(r"/devlogin", DevelopmentLoginHandler, name="devlogin"),
            url(r"/diff", DiffHandler, name="diff"),
            url(r"/download_assignment_scores/([^/]+)/([^/]+)", DownloadAssignmentScoresHandler, name="download_assignment_scores"),
            url(r"/download_file/([^/]+)/([^/]+)/([^/]+)/([^/]+)", DownloadFileHandler, name="download_file"),
            url(r"/download_submissions_student/([^/]+)", DownloadSubmissionsStudentHandler, name="download_submissions_student"),
            url(r"/edit_assignment/([^/]+)/([^/]+)?", EditAssignmentHandler, name="edit_assignment"),
            url(r"/edit_assignment_scores/([^/]+)/([^/]+)/([^/]+)", EditAssignmentScoresHandler, name="edit_assignment_scores"),
            url(r"/edit_course/([^/]+)?", EditCourseHandler, name="edit_course"),
            url(r"/edit_exercise/([^/]+)/([^/]+)/([^/]+)?", EditExerciseHandler, name="edit_exercise"),
            url(r"/exercise/([^/]+)/([^/]+)/([^/]+)", ExerciseHandler, name="exercise"),
            url(r"/exercise_feedback/([^/]+)/([^/]+)/([^/]+)", FeedbackHandler, name="exercise_feedback"),
            url(r"/exercise_hint_code/([^/]+)/([^/]+)/([^/]+)", HintCodeHandler, name="exercise_hint_code"),
            url(r"/exercise_pseudo_code/([^/]+)/([^/]+)/([^/]+)", PseudoHandler, name="exercise_pseudo_code"),
            url(r"/exercise_submissions/([^/]+)/([^/]+)/([^/]+)", ExerciseSubmissionsHandler, name="exercise_submissions"),
            url(r"/export_assignment/([^/]+)/([^/]+)", ExportAssignmentHandler, name="export_assignment"),
            url(r"/external/([^/]+)/([^/]+)/(.+)", ExternalSiteHandler, name="external"),
            url(r"/get_partner_id/([^/]+)/([^/]+)", GetPartnerIDHandler, name="get_partner_id"),
            url(r"/googlelogin", GoogleLoginHandler, name="googlelogin"),
            url(r"/help_requests/([^/]+)", HelpRequestsHandler, name="help_requests"),
            url(r"/import_assignment/([^/]+)", ImportAssignmentHandler, name="import_assignment"),
            url(r"/is_taking_restricted_assignment/([^/]+)/([^/]+)", IsTakingRestrictedAssignmentHandler, name="is_taking_restricted_assignment"),
            url(r"/login", LoginHandler, name="login"),
            url(r"/logout", LogoutHandler, name="logout"),
            url(r"/manage_admins", ManageAdminsHandler, name="manage_admins"),
            url(r"/manage_assistants/([^/]+)", ManageAssistantsHandler, name="manage_assistants"),
            url(r"/manage_instructors/([^/]+)", ManageInstructorsHandler, name="manage_instructors"),
            url(r"/manage_students/([^/]+)", ManageStudentsHandler, name="manage_students"),
            url(r"/manage_users", ManageUsersHandler, name="manage_users"),
            url(r"/move_assignment/([^/]+)/([^/]+)", MoveAssignmentHandler, name="move_assignment"),
            url(r"/move_exercise/([^/]+)/([^/]+)/([^/]+)?", MoveExerciseHandler, name="move_exercise"),
            # url(r"/profile/help_requests", ProfileHelpRequestsHandler, name="profile_help_requests"),
            url(r"/personal_info/([^/]+)", PersonalInfoHandler, name="personal_info"),
            url(r"/preferences/([^/]+)", PreferencesHandler, name="preferences"),
            # url(r"/profile/student_help_requests", ProfileStudentHelpRequestsHandler, name="profile_student_help_requests"),
            url(r"/register/([^/]+)/([^/]+)/([^/]+)", RegisterHandler, name="register"),
            url(r"/remove_admin", RemoveAdminHandler, name="remove_admin"),
            url(r"/remove_assistant/([^/]+)/([^/]+)", RemoveAssistantHandler, name="remove_assistant"),
            url(r"/remove_instructor/([^/]+)/([^/]+)", RemoveInstructorHandler, name="remove_instructor"),
            url(r"/resave_exercises/([^/]+)/([^/]+)", ResaveExercisesHandler, name="resave_exercises"),
            url(r"/reset_timer/([^/]+)/([^/]+)/([^/]+)", ResetTimerHandler, name="reset_timer"),
            url(r"/run_code/([^/]+)/([^/]+)/([^/]+)", RunCodeHandler, name="run_code"),
            url(r"/save_presubmission/([^/]+)/([^/]+)/([^/]+)", SavePresubmissionHandler, name="save_presubmission"),
            url(r"/static/(.+)", StaticFileHandler, name="static_file"),
            url(r"/student_exercise/([^/]+)/([^/]+)/([^/]+)/([^/]+)", StudentExerciseHandler, name="student_exercise"),
            url(r"/submit/([^/]+)/([^/]+)/([^/]+)", SubmitHandler, name="submit"),
            url(r"/submit_request/([^/]+)/([^/]+)/([^/]+)", SubmitHelpRequestHandler, name="submit_request"),
            url(r"/summarize_logs", SummarizeLogsHandler, name="summarize_logs"),
            url(r"/test", TestHandler, name="test"),
            url(r"/unregister/([^/]+)/([^/]+)", UnregisterHandler, name="unregister"),
            url(r"/unavailable_exercise/([^/]+)/([^/]+)", UnavailableExerciseHandler, name="unavailable_exercise"),
            url(r"/view_instructor_solution/([^/]+)/([^/]+)/([^/]+)", ViewInstructorSolutionHandler, name="view_instructor_solution"),
            url(r"/view_peer_solution/([^/]+)/([^/]+)/([^/]+)", ViewPeerSolutionHandler, name="view_peer_solution"),
            url(r"/view_request/([^/]+)/([^/]+)/([^/]+)/([^/]+)", ViewHelpRequestsHandler, name="view_request"),
            url(r"/view_assignment_scores/([^/]+)/([^/]+)", ViewAssignmentScoresHandler, name="view_assignment_scores"),
            url(r"/view_exercise_scores/([^/]+)/([^/]+)/([^/]+)", ViewExerciseScoresHandler, name="view_exercise_scores"),
            url(r"/view_student_assignment_scores/([^/]+)/([^/]+)", ViewStudentAssignmentScoresHandler, name="view_student_assignment_scores")
        ],
        default_handler_class=BaseHandler,
        autoescape=None,
        debug=(int(settings_dict["f_num_processes"]) == 1 and 'DEBUG' in os.environ and os.environ['DEBUG'] == 'true'),
        ui_methods=ui_methods
    )
    # Debugging doesn't seem to work on MacOS when running with two processes (https://github.com/tornadoweb/tornado/issues/2426)

    app.settings['template_path'] = os.path.join(os.path.dirname(__file__), "html")

    return app

class StaticFileHandler(BaseHandler):
    async def get(self, file_name):
        if file_name.endswith(".html"):
            try:
                self.render(file_name)
            except Exception as inst:
                render_error(self, traceback.format_exc())
        else:
            content_type = "text/css"
            read_mode = "r"

            if file_name.endswith(".js"):
                content_type = "text/javascript"
            elif file_name.endswith(".png"):
                content_type = "image/png"
                read_mode = "rb"
            elif file_name.endswith(".ico"):
                content_type = "image/x-icon"
                read_mode = "rb"
            elif file_name.endswith(".ttf"):
                content_type = "font/ttf"
                read_mode = "rb"
            elif file_name.endswith(".webmanifest"):
                content_type = "application/json"
            elif file_name.endswith(".json"):
                content_type = "application/json"

            file_contents = read_file("static/{}".format(file_name), mode=read_mode)

            self.set_header('Content-type', content_type)
            self.write(file_contents)

if __name__ == "__main__":
    try:
        settings_dict = load_yaml_dict(read_file("../Settings.yaml"))

        content = Content(settings_dict)

        database_version = content.get_database_version()
        code_version = int(read_file("../VERSION").rstrip())

        if database_version != code_version:
            print(f"Current database version: {database_version}")
            print(f"Current code version: {code_version}")

        # Check to see whether there is a database migration script (should only be one per version).
        # If so, make a backup copy of the database and then do the migration.
        for v in range(database_version, code_version):
            migration = f"{v}_to_{v + 1}"
            print(f"Checking database status for version {v+1}...")

            if os.path.isfile(f"migration_scripts/{migration}.py"):
                command = f"python migration_scripts/{migration}.py"
            else:
                command = f"python migration_scripts/migrate.py {migration}"

            result = run_command(command)

            if "***NotNeeded***" in result:
                print("Database migration not needed.")

                if ((v + 1) == code_version):
                    content.update_database_version(v + 1)
            elif "***Success***" in result:
                print(f"Database successfully migrated to version {v+1}")
                content.update_database_version(v + 1)
            else:
                print(f"Database migration failed for version {v+1}...")
                print(result)
                sys.exit(1)

        # Clean some stuff in the database
        content.delete_old_presubmissions()

        application = make_app(settings_dict)

        if settings_dict["mode"] == "development":
            server = tornado.httpserver.HTTPServer(application, max_header_size=1048576)
        else:
            server = tornado.httpserver.HTTPServer(application, max_header_size=1048576, ssl_options={
              "certfile": "/certs/cert.crt",
              "keyfile": "/certs/cert.key",
            })

        secrets_dict = load_yaml_dict(read_file("secrets/front_end.yaml"))
        application.settings["cookie_secret"] = secrets_dict["cookie"]
        application.settings["google_oauth"] = {
           "key": secrets_dict["google_oauth_key"],
           "secret": secrets_dict["google_oauth_secret"]
        }

        server.bind(int(settings_dict["f_port"]))
        server.start(int(settings_dict["f_num_processes"]))

        # Set up logging
        log_level = logging.INFO
        if settings_dict["mode"] == "development":
            log_level = logging.DEBUG

        log_file_handler = ConcurrentRotatingFileHandler("logs/codebuddy.log", maxBytes=100*1024*1024, backupCount=10, encoding="utf-8", mode="a")

        logging.basicConfig(
            handlers=[log_file_handler],
            level=log_level,
            format="[%(asctime)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger('codebuddy_logger')
        logger.addHandler(log_file_handler)

        logging.getLogger('tornado.access').disabled = True
        logging.getLogger("requests").setLevel(logging.DEBUG)

        logging.debug(f"Starting on port {settings_dict['f_port']} using {settings_dict['f_num_processes']} processes")
        tornado.ioloop.IOLoop.instance().start()
    except Exception as inst:
        print(traceback.format_exc())
        logging.error(traceback.format_exc())
        sys.exit(1)