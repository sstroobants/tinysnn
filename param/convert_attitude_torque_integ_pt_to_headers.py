from string import Template
import torch
import numpy as np

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights

MASK = True

if __name__ == "__main__": 
    # Load network
    attitude_state_dict = torch.load(f"param/models/model_attitude_worldly_forest.pt", map_location=torch.device('cpu'))
    # torque_state_dict = torch.load(f"param/models/model_torque_vague_pine.pt", map_location=torch.device('cpu'))
    # torque_state_dict = torch.load(f"param/models/model_torque_wandering_dew.pt", map_location=torch.device('cpu'))
    torque_state_dict = torch.load(f"param/models/model_torque_valiant_sunset.pt", map_location=torch.device('cpu'))

    if "l1.synapse_ff.weight" in torque_state_dict:
        rec = True
    else:
        rec = False

    attitude_layer_name = 'l2.synapse_ff.weight' if rec else 'l2.synapse.weight'
    torque_layer_name = 'l1.synapse_ff.weight' if rec else 'l1.synapse.weight'
    if MASK:
        attitude_mask = np.loadtxt('param/models/mask_worldly_forest.csv', delimiter=',')
        torque_mask = np.loadtxt('param/models/mask_valiant_sunset.csv', delimiter=',')
    else:
        attitude_mask = np.ones(attitude_state_dict[attitude_layer_name].size()[0])
        torque_mask = np.ones(torque_state_dict[torque_layer_name].size()[0])
    n_masked_neurons_torque = sum(torque_mask==0)
    n_masked_neurons_attitude = sum(attitude_mask==0)

    attitude_enc_size = attitude_state_dict["l1.neuron.leak_i"].size()[0]
    attitude_hidden_size = attitude_state_dict["l2.neuron.leak_i"].size()[0] - n_masked_neurons_attitude

    ############# -- CONTROLLER -- #################################

    if rec:
        controller_conf_params = {
            'input_size': attitude_state_dict["l1.synapse.weight"].size()[1] + 2,
            'encoding_size': attitude_state_dict["l1.synapse.weight"].size()[0],
            'hidden_size': attitude_hidden_size,
            'integ_size': 4,
            'hidden2_size': torque_state_dict["l1.synapse_ff.weight"].size()[0] - n_masked_neurons_torque,
            'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
            'type': 1,
        }
    else:
        controller_conf_params = {
            'input_size': attitude_state_dict["l1.synapse.weight"].size()[1] + 2,
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
    new_weights = new_weights * 0.001
    create_connection_from_template_with_weights('hidinteg', new_weights)

    ################### test_controller_integ_file
    create_softreset_integrator_from_template('integ')


    ################### test_controller_hidhid2_file
    N = controller_conf_params['hidden2_size']
    M = controller_conf_params['hidden_size']
    
    new_weights = torch.zeros([N, M + 4])
    if rec:
        torque_masked_weights = torque_state_dict['l1.synapse_ff.weight'][torch.tensor(torque_mask).type(torch.float) == 1, :]
        new_weights[:, :4] = torque_masked_weights[:, :4]
        new_weights[:, 4:] = torch.mm(torque_masked_weights[:, 4:], attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1])
    else:
        torque_masked_weights = torque_state_dict['l1.synapse_ff.weight'][torch.tensor(torque_mask).type(torch.float) == 1, :]
        new_weights[:, :4] = torque_masked_weights[:, :4]
        new_weights[:, 4:] = torch.mm(torque_masked_weights[:, 4:], attitude_state_dict['p_out.synapse.weight'][:, torch.tensor(attitude_mask).type(torch.float) == 1])

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