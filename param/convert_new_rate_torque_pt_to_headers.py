from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights


if __name__ == "__main__": 
    # Load network
    rate_state_dict = torch.load(f"param/models/model_rate_decent_snowball.pt") # gyro combination
    torque_state_dict = torch.load(f"param/models/model_torque_pleasant_morning.pt")

    # rate_hidden_size = rate_state_dict["enc.neuron.leak_i"].size()[0]
    rate_hidden_size = rate_state_dict["l1.neuron.leak_i"].size()[0]

    ############# -- CONTROLLER -- #################################

    controller_conf_params = {
        'input_size': rate_state_dict["l1.synapse.weight"].size()[1],
        'encoding_size': rate_state_dict["l1.synapse.weight"].size()[0],
        'hidden_size': rate_state_dict["l2.synapse_ff.weight"].size()[0],
        'integ_size': 4,
        # 'hidden2_size': torque_state_dict["rec.ff.weight"].size()[0],
        'hidden2_size': torque_state_dict["l1.synapse_ff.weight"].size()[0],
        # 'output_size': torque_state_dict["readout.weight"].size()[0],
        'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
        'type': 1,
    }

    controller_conf_template = 'param/templates/test_controller_ratetorqueinteg_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    create_connection_from_template('inenc', rate_state_dict, 'l1.synapse.weight')

    ################### test_controller_enc_file
    create_neuron_from_template('enc', rate_state_dict, 'l1.neuron', sigmoid=True)

    ################### test_controller_enchid_file
    create_connection_from_template('enchid', rate_state_dict, 'l2.synapse_ff.weight')

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', rate_state_dict, 'l2.synapse_rec.weight')

    ################### test_controller_hid_file
    create_neuron_from_template('hid', rate_state_dict, 'l2.neuron', sigmoid=True)

    ################### test_controller_hidinteg_file
    N = controller_conf_params['integ_size']
    M = controller_conf_params['hidden_size']
    new_weights = torch.zeros([N, M])
    integ_weights = torch.tensor([[1, 0, -(5 / 3), 0],
                                  [-1, 0, (5 / 3), 0],
                                  [0, 1, 0, (5 / 3)],
                                  [0, -1, 0, -(5 / 3)]], dtype=torch.float) * 0.003
    new_weights = torch.mm(integ_weights, rate_state_dict['p_out.synapse.weight'])
    create_connection_from_template_with_weights('hidinteg', new_weights)

    ################### test_controller_integ_file
    create_softreset_integrator_from_template('integ')


    ################### test_controller_hidhid2_file
    N = controller_conf_params['hidden2_size']
    M = controller_conf_params['hidden_size']
    # new_weights = torch.zeros([N, M + 2])
    new_weights = torch.zeros([N, M])
    # Rate state dict needs to be reordered to match correct input order. If retrained, this can be fixed. 
    new_weights = torch.mm(torque_state_dict['l1.synapse_ff.weight'], rate_state_dict['p_out.synapse.weight'])
    # new_weights = torch.mm(torque_state_dict['l1.synapse_ff.weight'], rate_state_dict['p_out.synapse.weight'][[2, 3, 0, 1], :])
    # new_weights = torch.mm(torque_state_dict['rec.ff.weight'], rate_state_dict['readout.weight'][[2, 3, 0, 1], :])
    # new_weights[[0, 1], :] = new_weights[[0, 1], :] * 3
    create_connection_from_template_with_weights('hidhid2', new_weights) 

    ################### test_controller_hid2hid2_file
    # create_connection_from_template('hid2hid2', torque_state_dict, 'rec.rec.weight')
    create_connection_from_template('hid2hid2', torque_state_dict, 'l1.synapse_rec.weight')

    ################### test_controller_hid2_file
    # create_neuron_from_template('hid2', torque_state_dict, 'rec.neuron', sigmoid=False)
    create_neuron_from_template('hid2', torque_state_dict, 'l1.neuron', sigmoid=True)

    ################### test_controller_hid2out_file
    # create_connection_from_template('hid2out', torque_state_dict, 'readout.weight')
    create_connection_from_template('hid2out', torque_state_dict, 'p_out.synapse.weight')

    ################### test_controller_li_out

    li_out_params = {
        # 'leak': f"{torque_state_dict['p_out.neuron.leak_v'][0].item()}",
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)