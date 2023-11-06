from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights


if __name__ == "__main__": 
    # Load network
    rate_state_dict = torch.load(f"param/models/model_rate_gyro_controller.pt") # gyro combination
    torque_state_dict = torch.load(f"param/models/model_torque_controller.pt")

    rate_hidden_size = rate_state_dict["enc.neuron.leak_i"].size()[0]

    ############# -- CONTROLLER -- #################################

    controller_conf_params = {
        'input_size': rate_state_dict["enc.ff.weight"].size()[1],
        'encoding_size': rate_state_dict["enc.ff.weight"].size()[0],
        'hidden_size': rate_state_dict["rec.ff.weight"].size()[0],
        'hidden2_size': torque_state_dict["rec.ff.weight"].size()[0],
        'output_size': torque_state_dict["readout.weight"].size()[0],
        'type': 1,
    }
    controller_conf_template = 'param/templates/test_controller_ratetorque_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    # combine the weights from rate and torque
    # new_weights = 
    create_connection_from_template('inenc', rate_state_dict, 'enc.ff.weight')

    ################### test_controller_enc_file
    create_neuron_from_template('enc', rate_state_dict, 'enc.neuron')

    ################### test_controller_enchid_file
    create_connection_from_template('enchid', rate_state_dict, 'rec.ff.weight')

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', rate_state_dict, 'rec.rec.weight')

    ################### test_controller_hid_file
    create_neuron_from_template('hid', rate_state_dict, 'rec.neuron')

    ################### test_controller_hidhid2_file
    N = controller_conf_params['hidden2_size']
    M = controller_conf_params['hidden_size']
    # new_weights = torch.zeros([N, M + 2])
    new_weights = torch.zeros([N, M])
    new_weights = torch.mm(torque_state_dict['rec.ff.weight'], rate_state_dict['readout.weight'])
    # new_weights[:, 2:] = torch.mm(torque_state_dict['rec.ff.weight'][:, 2:], rate_state_dict['readout.weight'])
    # new_weights[:, :2] = torque_state_dict['rec.ff.weight'][:, :2]
    create_connection_from_template_with_weights('hidhid2', new_weights)

    ################### test_controller_hid2hid2_file
    create_connection_from_template('hid2hid2', torque_state_dict, 'rec.rec.weight')

    ################### test_controller_hid2_file
    create_neuron_from_template('hid2', torque_state_dict, 'rec.neuron')

    ################### test_controller_hid2out_file
    create_connection_from_template('hid2out', torque_state_dict, 'readout.weight')

    ################### test_controller_li_out

    li_out_params = {
        # 'leak': f"{torque_state_dict['p_out.neuron.leak_v'][0].item()}",
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)