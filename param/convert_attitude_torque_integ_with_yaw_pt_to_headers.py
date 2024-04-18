import torch
import numpy as np
import os

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights

MASK = True

if __name__ == "__main__": 
    # Load network
    # att_name = "absurd-snow-174"
    # att_name = "distinctive-capybara-169"
    att_name = "abundant-moon-184"
    # att_name = "lambent-paper-185"
    # att_name = "alight-ox-189"
    
    torque_name = "warm-armadillo-179"
    # torque_name = "lucky-paper-1
    # torque_name = "burning-darling-192"

    dirname = os.getcwd()
    att_folder = os.path.abspath(os.path.join(dirname, f"../ultrasonic-snn/runs/{att_name}"))
    attitude_state_dict = torch.load(os.path.join(att_folder, "model.pt"), map_location=torch.device('cpu'))
    torque_folder = os.path.abspath(os.path.join(dirname, f"../ultrasonic-snn/runs/{torque_name}"))
    torque_state_dict = torch.load(os.path.join(torque_folder, "model.pt"), map_location=torch.device('cpu'))

    if "l1.synapse_ff.weight" in torque_state_dict:
        rec = True
    else:
        rec = False

    attitude_enc_layer_name = "l1.synapse.weight"
    attitude_layer_name = 'l2.synapse_ff.weight' if rec else 'l2.synapse.weight'
    torque_layer_name = 'l1.synapse_ff.weight' if rec else 'l1.synapse.weight'
    if MASK:
        attitude_mask = np.loadtxt(os.path.join(att_folder, "mask_2.csv"), delimiter=',')
        torque_mask = np.loadtxt(os.path.join(torque_folder, "mask.csv"), delimiter=',')
        # torque_mask = np.ones(torque_state_dict[torque_layer_name].size()[0])
    else:
        attitude_mask = np.ones(attitude_state_dict[attitude_layer_name].size()[0])
        torque_mask = np.ones(torque_state_dict[torque_layer_name].size()[0])
    n_masked_neurons_torque = sum(torque_mask==0)
    n_masked_neurons_attitude = sum(attitude_mask==0)

    print(f"\nAttitude network: {att_name}")
    print(f"{attitude_state_dict[attitude_enc_layer_name].size()[0]} neurons attitude network first layer")
    print(f"{len(attitude_mask) - n_masked_neurons_attitude} neurons attitude network second layer ([{n_masked_neurons_attitude}/{len(attitude_mask)}] masked)\n")
    print(f"Torque network: {torque_name}")
    print(f"{len(torque_mask) - n_masked_neurons_torque} neurons in the torque network ([{n_masked_neurons_torque}/{len(torque_mask)}] masked)\n")

    n_l1 = attitude_state_dict[attitude_enc_layer_name].size()[0]
    n_l2 = len(attitude_mask) - n_masked_neurons_attitude
    n_l3 = len(torque_mask) - n_masked_neurons_torque

    print(f"Total multiplications approx {6*n_l1} + {n_l1*n_l2} + {n_l2*n_l2} + {(n_l2*n_l3) + (2*n_l3)} + {n_l3*n_l3} + {n_l3*3} = {(6*n_l1) + (n_l1*n_l2) + (n_l2*n_l2) + (n_l2*n_l3) + (n_l3*n_l3) + (2*n_l3) + (n_l3*3)}")
    attitude_enc_size = attitude_state_dict["l1.neuron.leak_i"].size()[0]
    attitude_hidden_size = attitude_state_dict["l2.neuron.leak_i"].size()[0] - n_masked_neurons_attitude

    ############# -- CONTROLLER -- #################################

    if rec:
        controller_conf_params = {
            'input_size': attitude_state_dict["l1.synapse.weight"].size()[1] + 3,
            'encoding_size': attitude_state_dict["l1.synapse.weight"].size()[0],
            'hidden_size': attitude_hidden_size,
            'integ_size': 4,
            'hidden2_size': torque_state_dict["l1.synapse_ff.weight"].size()[0] - n_masked_neurons_torque,
            'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
            'type': 1,
        }
    else:
        controller_conf_params = {
            'input_size': attitude_state_dict["l1.synapse.weight"].size()[1] + 3,
            'encoding_size': attitude_state_dict["l1.synapse.weight"].size()[0],
            'hidden_size': attitude_hidden_size,
            'integ_size': 4,
            'hidden2_size': torque_state_dict["l1.synapse.weight"].size()[0] - n_masked_neurons_torque,
            'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
            'type': 1,
        }

    controller_conf_template = 'param/templates/test_controller_ratetorqueinteg_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    create_connection_from_template('inenc', attitude_state_dict, 'l1.synapse.weight')

    ################### test_controller_enc_file
    create_neuron_from_template('enc', attitude_state_dict, 'l1.neuron', sigmoid=True)

    ################### test_controller_enchid_file
    create_connection_from_template('enchid', attitude_state_dict, 'l2.synapse_ff.weight', mask1=attitude_mask, mask2=None)

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', attitude_state_dict, 'l2.synapse_rec.weight', mask1=attitude_mask, mask2=attitude_mask)

    ################### test_controller_hid_file
    create_neuron_from_template('hid', attitude_state_dict, 'l2.neuron', sigmoid=True, mask=attitude_mask)

    ################### test_controller_hidinteg_file
    N = controller_conf_params['integ_size']
    M = controller_conf_params['hidden_size']
    new_weights = torch.zeros([N, M+2])
    new_weights[:2, :M] = attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1]
    new_weights[2:, :M] = -attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1]
    new_weights[:, M:] = torch.tensor([[0, -(5/3)],
                                        [-(5/3), 0],
                                        [0, (5/3)],
                                        [(5/3), 0]], dtype=torch.float)
    new_weights = new_weights * 0.005
    create_connection_from_template_with_weights('hidinteg', new_weights)

    ################### test_controller_integ_file
    create_softreset_integrator_from_template('integ')


    ################### test_controller_hidhid2_file
    N = controller_conf_params['hidden2_size']
    M = controller_conf_params['hidden_size']
    
    n_extra = 6
    new_weights = torch.zeros([N, M + n_extra])
    if rec:
        torque_masked_weights = torque_state_dict['l1.synapse_ff.weight'][torch.tensor(torque_mask).type(torch.float) == 1, :]
        new_weights[:, :n_extra] = torque_masked_weights[:, :n_extra]
        new_weights[:, n_extra:] = torch.mm(torque_masked_weights[:, n_extra:], attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1])
    else:
        torque_masked_weights = torque_state_dict['l1.synapse_ff.weight'][torch.tensor(torque_mask).type(torch.float) == 1, :]
        new_weights[:, :n_extra] = torque_masked_weights[:, :n_extra]
        new_weights[:, n_extra:] = torch.mm(torque_masked_weights[:, n_extra:], attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1])

    create_connection_from_template_with_weights('hidhid2', new_weights)

    ################### test_controller_hid2hid2_file
    if rec:
        create_connection_from_template('hid2hid2', torque_state_dict, 'l1.synapse_rec.weight', mask1=torque_mask, mask2=torque_mask)

    ################### test_controller_hid2_file
    create_neuron_from_template('hid2', torque_state_dict, 'l1.neuron', sigmoid=True, mask=torque_mask)

    ################### test_controller_hid2out_file
    create_connection_from_template('hid2out', torque_state_dict, 'p_out.synapse.weight', mask2=torque_mask)

    ################### test_controller_li_out

    li_out_params = {
        # 'leak': f"{torque_state_dict['p_out.neuron.leak_v'][0].item()}",
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)