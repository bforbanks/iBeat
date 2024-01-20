def convert_to_list(obj, property_name):
    if type(getattr(obj, property_name)) is not list:
        setattr(obj, property_name, [getattr(obj, property_name)])
