from string import Template
import torch

def create_from_template(template_filename, output_filename, params):
    '''
    Take template file and the parameters to be exchanged, fill in and write to file.
    '''
    with open(template_filename, 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)
        # print(result)
        with open(output_filename, 'w') as f_out:
            f_out.write(result)


# Load network
state_dict = torch.load(f"param/models/model.pt")
for name in state_dict.keys():
    print(name)
hidden_size = state_dict["proportional.l1.neuron.leak_i"].size()[0]
hidden_size_deriv = state_dict["derivative.l1.neuron.leak_i"].size()[0] + state_dict["derivative.l2.neuron.leak_i"].size()[0]
state_dict['alpha_param'] = 0.3
state_dict['beta_param'] = 0.9
print('ALPHA AND BETA ARE NOT YET TAKEN FROM MODEL')

################### test_pid_conf
pid_conf_params = {
    'p_gain': f'{state_dict["p_gain"].item():.5f}f',
    'i_gain': f'{state_dict["i_gain"].item():.5f}f',
    'd_gain': f'{state_dict["d_gain"].item():.5f}f',
    'type': 'prop'
}
pid_conf_template = 'param/templates/test_pid_conf.templ'
pid_conf_out = 'param/test_pid_conf.h'

create_from_template(pid_conf_template, pid_conf_out, pid_conf_params)

############# -- PROPORTIONAL -- #################################
################### test_proportional_conf
prop_conf_params = {
    'hidden_size': f'{hidden_size}',
    'type': 'prop'
}
prop_conf_template = 'param/templates/test_conf.templ'
prop_conf_out = 'param/proportional/test_prop_conf.h'

create_from_template(prop_conf_template, prop_conf_out, prop_conf_params)

################### test_proportional_enc_file
prop_enc_params = {
    'hidden_size': f'{hidden_size}',
    'alpha': f"{state_dict['alpha_param']:.2f}f",
    'beta': f"{state_dict['beta_param']:.2f}f",
    'type': 'prop'
}
prop_enc_template = 'param/templates/test_enc_file.templ'
prop_enc_out = 'param/proportional/test_prop_enc_file.h'

create_from_template(prop_enc_template, prop_enc_out, prop_enc_params)

################### test_proportional_hid_file
d_i_string = '{'
d_v_string = '{'
add_thresh = '{'
for i in range(hidden_size):
    d_i_string += f"{state_dict['proportional.l1.neuron.leak_i'][i].item():2f}f, "
    d_v_string += f"{state_dict['proportional.l1.neuron.leak_v'][i].item():2f}f, "
    add_thresh += f"{0:.2f}f, "
d_i_string = d_i_string[:-2] + '}'
d_v_string = d_v_string[:-2] + '}'
add_thresh = add_thresh[:-2] + '}'
prop_hid_params = {
    'hidden_size': f'{hidden_size}',
    'd_i': f"{d_i_string}",
    'd_v': f"{d_v_string}",
    'add_thresh': add_thresh,
    'type': 'prop'
}
prop_hid_template = 'param/templates/test_hid_file.templ'
prop_hid_out = 'param/proportional/test_prop_hid_file.h'

create_from_template(prop_hid_template, prop_hid_out, prop_hid_params)

################### test_proportional_hidout_file
w_hidout_string = '{'
for i in range(state_dict['proportional.prop_output_weights'].size()[0]):
    w_hidout_string += f"{state_dict['proportional.prop_output_weights'][i].item():2f}f, "
    w_hidout_string += f"{-state_dict['proportional.prop_output_weights'][i].item():2f}f, "
w_hidout_string = w_hidout_string[:-2] + '}'

prop_hidout_params = {
    'hidden_size': f'{hidden_size}',
    'w_hidout': f"{w_hidout_string}",
    'type': 'prop'
}
prop_hidout_template = 'param/templates/test_hidout_file.templ'
prop_hidout_out = 'param/proportional/test_prop_hidout_file.h'

create_from_template(prop_hidout_template, prop_hidout_out, prop_hidout_params)

################### test_proportional_inhid_file
w_inhid_string = '{'
for i in range(state_dict['proportional.prop_input_weights'].size()[0]):
    w_inhid_string += f"{state_dict['proportional.prop_input_weights'][i].item():2f}f, "
    w_inhid_string += f"{state_dict['proportional.prop_input_weights'][i].item():2f}f, "
w_inhid_string = w_inhid_string[:-2] + '}'

prop_inhid_params = {
    'hidden_size': f'{hidden_size}',
    'w_inhid': f"{w_inhid_string}",
    'type': 'prop'
}
prop_inhid_template = 'param/templates/test_inhid_file.templ'
prop_inhid_out = 'param/proportional/test_prop_inhid_file.h'

create_from_template(prop_inhid_template, prop_inhid_out, prop_inhid_params)

################### test_prop_li_out_file

prop_li_out_params = {
    'leak': f"{state_dict['proportional.p_out.neuron.leak_v'][0].item()}",
    'type': 'prop'
}
prop_li_out_template = 'param/templates/test_li_out_file.templ'
prop_li_out_out = 'param/proportional/test_prop_li_out_file.h'

create_from_template(prop_li_out_template, prop_li_out_out, prop_li_out_params)

############# -- INTEGRAL -- #################################
################### test_integral_conf
integ_conf_params = {
    'hidden_size': f'{hidden_size}',
    'type': 'integ'
}
integ_conf_template = 'param/templates/test_conf.templ'
integ_conf_out = 'param/integral/test_integ_conf.h'

create_from_template(integ_conf_template, integ_conf_out, integ_conf_params)

################### test_integral_enc_file
integ_enc_params = {
    'hidden_size': f'{hidden_size}',
    'alpha': f"{state_dict['alpha_param']:.2f}f",
    'beta': f"{state_dict['beta_param']:.2f}f",
    'type': 'integ'
}
integ_enc_template = 'param/templates/test_enc_file.templ'
integ_enc_out = 'param/integral/test_integ_enc_file.h'

create_from_template(integ_enc_template, integ_enc_out, integ_enc_params)

################### test_integral_hid_file
d_i_string = '{'
d_v_string = '{'
add_thresh = '{'
for i in range(hidden_size):
    d_i_string += f"{state_dict['integral.l1.neuron.leak_i'][i].item():2f}f, "
    d_v_string += f"{state_dict['integral.l1.neuron.leak_v'][i].item():2f}f, "
    add_thresh += f"{state_dict['integral.l1.neuron.add_t'][i].item():2f}f, "
d_i_string = d_i_string[:-2] + '}'
d_v_string = d_v_string[:-2] + '}'
add_thresh = add_thresh[:-2] + '}'
integ_hid_params = {
    'hidden_size': f'{hidden_size}',
    'd_i': f"{d_i_string}",
    'd_v': f"{d_v_string}",
    'add_thresh': add_thresh,
    'type': 'integ'
}
integ_hid_template = 'param/templates/test_hid_file.templ'
integ_hid_out = 'param/integral/test_integ_hid_file.h'

create_from_template(integ_hid_template, integ_hid_out, integ_hid_params)

################### test_integral_hidout_file
w_hidout_string = '{'
for i in range(state_dict['integral.integ_output_weights'].size()[0]):
    w_hidout_string += f"{state_dict['integral.integ_output_weights'][i].item():2f}f, "
    w_hidout_string += f"{-state_dict['integral.integ_output_weights'][i].item():2f}f, "
w_hidout_string = w_hidout_string[:-2] + '}'

integ_hidout_params = {
    'hidden_size': f'{hidden_size}',
    'w_hidout': f"{w_hidout_string}",
    'type': 'integ'
}
integ_hidout_template = 'param/templates/test_hidout_file.templ'
integ_hidout_out = 'param/integral/test_integ_hidout_file.h'

create_from_template(integ_hidout_template, integ_hidout_out, integ_hidout_params)

################### test_integral_inhid_file
w_inhid_string = '{'
for i in range(state_dict['integral.integ_input_weights'].size()[0]):
    w_inhid_string += f"{state_dict['integral.integ_input_weights'][i].item():2f}f, "
    w_inhid_string += f"{state_dict['integral.integ_input_weights'][i].item():2f}f, "
w_inhid_string = w_inhid_string[:-2] + '}'

integ_inhid_params = {
    'hidden_size': f'{hidden_size}',
    'w_inhid': f"{w_inhid_string}",
    'type': 'integ'
}
integ_inhid_template = 'param/templates/test_inhid_file.templ'
integ_inhid_out = 'param/integral/test_integ_inhid_file.h'

create_from_template(integ_inhid_template, integ_inhid_out, integ_inhid_params)

################### test_integ_li_out_file

integ_li_out_params = {
    'leak': f"{state_dict['integral.i_out.neuron.leak_v'][0].item()}",
    'type': 'integ'
}
integ_li_out_template = 'param/templates/test_li_out_file.templ'
integ_li_out_out = 'param/integral/test_integ_li_out_file.h'

create_from_template(integ_li_out_template, integ_li_out_out, integ_li_out_params)

############# -- DERIVATIVE -- #################################
################### test_derivative_conf
deriv_conf_params = {
    'hidden_size': f'{hidden_size_deriv}',
    'type': 'deriv'
}
deriv_conf_template = 'param/templates/test_conf.templ'
deriv_conf_out = 'param/derivative/test_deriv_conf.h'

create_from_template(deriv_conf_template, deriv_conf_out, deriv_conf_params)

################### test_derivative_enc_file
deriv_enc_params = {
    'hidden_size': f'{hidden_size_deriv}',
    'alpha': f"{state_dict['alpha_param']:.2f}f",
    'beta': f"{state_dict['beta_param']:.2f}f",
    'type': 'deriv'
}
deriv_enc_template = 'param/templates/test_enc_file.templ'
deriv_enc_out = 'param/derivative/test_deriv_enc_file.h'

create_from_template(deriv_enc_template, deriv_enc_out, deriv_enc_params)

################### test_derivative_hid_file
d_i_string = '{'
d_v_string = '{'
add_thresh = '{'
for i in range(int(hidden_size_deriv / 2)):
    d_i_string += f"{state_dict['derivative.l1.neuron.leak_i'][i].item():2f}f, "
    d_v_string += f"{state_dict['derivative.l1.neuron.leak_v'][i].item():2f}f, "
    add_thresh += f"{0:.2f}f, "
for i in range(int(hidden_size_deriv / 2)):
    d_i_string += f"{state_dict['derivative.l2.neuron.leak_i'][i].item():2f}f, "
    d_v_string += f"{state_dict['derivative.l2.neuron.leak_v'][i].item():2f}f, "
    add_thresh += f"{0:.2f}f, "
d_i_string = d_i_string[:-2] + '}'
d_v_string = d_v_string[:-2] + '}'
add_thresh = add_thresh[:-2] + '}'
deriv_hid_params = {
    'hidden_size': f'{hidden_size_deriv}',
    'd_i': f"{d_i_string}",
    'd_v': f"{d_v_string}",
    'add_thresh': add_thresh,
    'type': 'deriv'
}
deriv_hid_template = 'param/templates/test_hid_file.templ'
deriv_hid_out = 'param/derivative/test_deriv_hid_file.h'

create_from_template(deriv_hid_template, deriv_hid_out, deriv_hid_params)

################### test_derivative_hidout_file
w_hidout_string = '{'
for i in range(int(hidden_size_deriv / 4)):
    w_hidout_string += f"{state_dict['derivative.deriv_output_weights'][i].item():2f}f, "
    w_hidout_string += f"{-state_dict['derivative.deriv_output_weights'][i].item():2f}f, "
for i in range(int(hidden_size_deriv / 4), int(hidden_size_deriv / 2)):
    w_hidout_string += f"{-state_dict['derivative.deriv_output_weights'][i].item():2f}f, "
    w_hidout_string += f"{state_dict['derivative.deriv_output_weights'][i].item():2f}f, "
w_hidout_string = w_hidout_string[:-2] + '}'
deriv_hidout_params = {
    'hidden_size': f'{hidden_size_deriv}',
    'w_hidout': f"{w_hidout_string}",
    'type': 'deriv'
}
deriv_hidout_template = 'param/templates/test_hidout_file.templ'
deriv_hidout_out = 'param/derivative/test_deriv_hidout_file.h'

create_from_template(deriv_hidout_template, deriv_hidout_out, deriv_hidout_params)

################### test_derivative_inhid_file
w_inhid_string = '{'
for i in range(state_dict['derivative.fast_weights'].size()[0]):
    w_inhid_string += f"{state_dict['derivative.fast_weights'][i].item():2f}f, "
    w_inhid_string += f"{-state_dict['derivative.fast_weights'][i].item():2f}f, "
for i in range(state_dict['derivative.slow_weights'].size()[0]):
    w_inhid_string += f"{state_dict['derivative.slow_weights'][i].item():2f}f, "
    w_inhid_string += f"{-state_dict['derivative.slow_weights'][i].item():2f}f, "
w_inhid_string = w_inhid_string[:-2] + '}'

deriv_inhid_params = {
    'hidden_size': f'{hidden_size_deriv}',
    'w_inhid': f"{w_inhid_string}",
    'type': 'deriv'
}
deriv_inhid_template = 'param/templates/test_inhid_file.templ'
deriv_inhid_out = 'param/derivative/test_deriv_inhid_file.h'

create_from_template(deriv_inhid_template, deriv_inhid_out, deriv_inhid_params)

################### test_deriv_li_out_file

deriv_li_out_params = {
    'leak': f"{state_dict['derivative.d_out.neuron.leak_v'][0].item()}",
    'type': 'deriv'
}
deriv_li_out_template = 'param/templates/test_li_out_file.templ'
deriv_li_out_out = 'param/derivative/test_deriv_li_out_file.h'

create_from_template(deriv_li_out_template, deriv_li_out_out, deriv_li_out_params)