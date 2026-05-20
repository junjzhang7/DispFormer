import logging
import os

logger = logging.getLogger(__name__)


def set_logger(
    log_dir,
    displaying=True,
    saving=True,
    debug=False,
    wandb_logger=None,
    mlflow_logger=None,
):
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        saving = False

    logger = logging.getLogger()  # get root logger
    logger.handlers = []
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = None
    if displaying:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if saving:
        log_file_path = os.path.abspath(f"{log_dir}/run.log")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if wandb_logger:

        class WandbLogFilter(logging.Filter):
            def filter(self, record):
                return "wandb" not in record.name

        class WandbHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                wandb_logger.experiment.log({"logs": log_entry})

        handler = WandbHandler()
        handler.addFilter(WandbLogFilter())
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if mlflow_logger:
        pass

    return logger


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        logger.info(arg + "." * (str_num - len(arg) - len(str(val))) + str(val))
