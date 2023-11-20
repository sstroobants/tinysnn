from string import Template
import torch

def create_from_template(template_filename, output_filename, params):
    '''
    Take template file and the parameters to be exchanged, fill in and write to file.
    '''
    # pass
    with open(template_filename, 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)
        # print(result)
        with open(output_filename, 'w') as f_out:
            f_out.write(result)

def create_connection_from_template(name, state_dict, state_name):
    weights = state_dict[state_name]
    create_connection_from_template_with_weights(name, weights)


def create_connection_from_template_with_weights(name, weights):
    input_size = weights.size()[1]
    output_size = weights.size()[0]
    weights_string = '{'
    for i in range(weights.size()[0]):
        for j in range(weights.size()[1]):
            weights_string += f"{weights[i, j].item():2f}f, "
    weights_string = weights_string[:-2] + '}'

    params = {
        'input_size': f'{input_size}',
        'output_size': f'{output_size}',
        'weights': f'{weights_string}',
        'name': name
    }
    template = 'param/templates/test_connection_file.templ'
    out = f'param/controller/test_controller_{name}_file.h'

    create_from_template(template, out, params)

def create_neuron_from_template(name, state_dict, state_name):
    hidden_size = state_dict[f"{state_name}.leak_i"].size()[0]
    d_i_string = '{'
    d_v_string = '{'
    for i in range(hidden_size):
        leak_i = torch.clamp(state_dict[f"{state_name}.leak_i"][i], 0.0, 1.0).item()
        leak_v = torch.clamp(state_dict[f"{state_name}.leak_v"][i], 0.0, 1.0).item()
        d_i_string += f"{leak_i:2f}f, "
        d_v_string += f"{leak_v:2f}f, "
    d_i_string = d_i_string[:-2] + '}'
    d_v_string = d_v_string[:-2] + '}'
    params = {
        'name': name,
        'hidden_size': f'{hidden_size}',
        'd_i': f"{d_i_string}",
        'd_v': f"{d_v_string}",
    }
    template = 'param/templates/test_neuron_file.templ'
    out = f'param/controller/test_controller_{name}_file.h'

    create_from_template(template, out, params)

def create_softreset_integrator_from_template(name):
    hidden_size = 4
    d_i_string = '{'
    d_v_string = '{'
    for i in range(hidden_size):
        leak_i = 1.0
        leak_v = 1.0
        d_i_string += f"{leak_i:2f}f, "
        d_v_string += f"{leak_v:2f}f, "
    d_i_string = d_i_string[:-2] + '}'
    d_v_string = d_v_string[:-2] + '}'
    params = {
        'name': name,
        'hidden_size': f'{hidden_size}',
        'type': '2',
        'd_i': f"{d_i_string}",
        'd_v': f"{d_v_string}",
    }