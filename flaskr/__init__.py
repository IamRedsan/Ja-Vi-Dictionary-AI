def create_app():

    from flask import Flask

    app = Flask(__name__, instance_relative_config=True)

    from dotenv import load_dotenv, dotenv_values

    load_dotenv()
    app.config.from_mapping(dotenv_values())

    # json converter
    from flaskr.utils.json_helper import CustomJSONProvider

    app.json = CustomJSONProvider(app)

    # error handler
    from flaskr.errors.bad_request import BadRequestError
    from flaskr.errors.not_found import NotFoundError
    from flaskr.errors.unauthenicated import UnauthenticatedError
    from flaskr.errors.forbidden import ForbiddenError
    from werkzeug.exceptions import HTTPException
    from flaskr.errorHandlers.my_handler import my_handler
    from flaskr.errorHandlers.default_http_handler import default_http_handler
    from flaskr.errorHandlers.default_handler import default_handler

    app.register_error_handler(BadRequestError, my_handler)
    app.register_error_handler(NotFoundError, my_handler)
    app.register_error_handler(UnauthenticatedError, my_handler)
    app.register_error_handler(ForbiddenError, my_handler)
    app.register_error_handler(HTTPException, default_http_handler)
    app.register_error_handler(Exception, default_handler)

    # api/v1/translate
    from flaskr.controllers.translateController import translateBP

    app.register_blueprint(translateBP)

    # translate socket
    from flaskr.utils.translate_helper import TranslateManager

    TranslateManager.get_instance().init_app(app)
    TranslateManager.get_instance().start()

    import cloudinary

    cloudinary.config(
        cloud_name=app.config.get("CLOUD_NAME"),
        api_key=app.config.get("API_KEY"),
        api_secret=app.config.get("API_SECRET"),
        secure=True,
    )

    return app
