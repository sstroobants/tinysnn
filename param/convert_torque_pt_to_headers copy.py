import os
import torch
import numpy as np

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights


MASK = True 

if __name__ == "__main__": 
    # Load network
    torque_name = "warm-armadillo-179"
    dirname = os.getcwd()
    torque_folder = os.path.abspath(os.path.join(dirname, f"../ultrasonic-snn/runs/{torque_name}"))
    torque_state_dict = torch.load(os.path.join(torque_folder, "model.pt"), map_location=torch.device('cpu'))

    if "l1.synapse_ff.weight" in torque_state_dict:
        rec = True
    else:
        rec = False

    torque_layer_name = 'l1.synapse_ff.weight' if rec else 'l1.synapse.weight'
    if MASK:
        torque_mask = np.loadtxt(os.path.join(torque_folder, "mask.csv"), delimiter=',')
    else:
        torque_mask = np.ones(torque_state_dict[torque_layer_name].size()[0])
    n_masked_neurons_torque = sum(torque_mask==0)

    print(f"Torque network: {torque_name}")
    print(f"{len(torque_mask) - n_masked_neurons_torque} neurons in the torque network ([{n_masked_neurons_torque}/{len(torque_mask)}] masked)\n")

    ############# -- CONTROLLER -- #################################
    if rec:
        controller_conf_params = {
            'input_size': torque_state_dict["l1.synapse_ff.weight"].size()[1],
            'hidden_size': torque_state_dict["l1.synapse_ff.weight"].size()[0] - n_masked_neurons_torque,
            'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
            'type': 1,
        }
    else:
        controller_conf_params = {
            'input_size': torque_state_dict["l1.synapse.weight"].size()[1],
            'hidden_size': torque_state_dict["l1.synapse.weight"].size()[0] - n_masked_neurons_torque,
            'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
            'type': 1,
        }

    controller_conf_template = 'param/templates/test_controller_torque_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    if rec:
        create_connection_from_template('inhid', torque_state_dict, 'l1.synapse_ff.weight', mask1=torque_mask, mask2=None)
    else:
        create_connection_from_template('inhid', torque_state_dict, 'l1.synapse.weight', mask1=torque_mask, mask2=None)

    ################### test_controller_hidhid_file
    if rec:
        create_connection_from_template('hidhid', torque_state_dict, 'l1.synapse_rec.weight', mask1=torque_mask, mask2=torque_mask)

    ################### test_controller_hid_file
    create_neuron_from_template('hid', torque_state_dict, 'l1.neuron', sigmoid=True, mask=torque_mask)

    ################### test_controller_hidout_file
    create_connection_from_template('hidout', torque_state_dict, 'p_out.synapse.weight', mask1=None, mask2=torque_mask)

    ################### test_controller_li_out

    li_out_params = {
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)