class Inference_data_model(object):
    model_name = str()
    model_path = str()
    handler_path = str()
    index_file_path = str()
    input_path = str()
    model_arch_path = str()
    extra_files = str()
    gpus = int()
    gen_folder = str()
    mar_filepath = str()
    ts_log_file = str()
    ts_log_config = str()
    ts_config_file = str()
    ts_model_store = str()
    dir_path = str()


def set_data_model(data, gpus, gen_folder, model_name="", model_path="", handler_path="", 
        classes="", model_arch_path="", extra_files="", mar_filepath=""):
    data_model = Inference_data_model()
    data_model.model_name = model_name
    data_model.model_path = model_path
    data_model.handler_path = handler_path
    data_model.index_file_path = classes
    data_model.input_path = data
    data_model.model_arch_path = model_arch_path
    data_model.extra_files = extra_files
    data_model.gpus = gpus
    data_model.gen_folder = gen_folder
    data_model.mar_filepath=mar_filepath

    return data_model

