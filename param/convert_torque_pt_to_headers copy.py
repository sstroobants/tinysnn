from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights


if __name__ == "__main__": 
    # Load network
    # torque_state_dict = torch.load(f"param/models/model_torque_devoted_wood.pt") # new data 2x
    # torque_state_dict = torch.load(f"param/models/model_torque_stilted_microwave.pt") # old data
    # torque_state_dict = torch.load(f"param/models/model_torque_jumping_breeze.pt") # old data shifted 6
    torque_state_dict = torch.load(f"param/models/model_torque_super_plant.pt") # new data shifted 6 2x
    # torque_state_dict = torch.load(f"param/models/model_torque_sparkling_glade.pt") # old data 2x

    ############# -- CONTROLLER -- #################################

    controller_conf_params = {
        'input_size': torque_state_dict["l1.synapse_ff.weight"].size()[1],
        'hidden_size': torque_state_dict["l1.synapse_ff.weight"].size()[0],
        'output_size': torque_state_dict["p_out.synapse.weight"].size()[0],
        'type': 1,
    }

    controller_conf_template = 'param/templates/test_controller_torque_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    create_connection_from_template('inhid', torque_state_dict, 'l1.synapse_ff.weight')

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', torque_state_dict, 'l1.synapse_rec.weight')

    ################### test_controller_hid_file
    create_neuron_from_template('hid', torque_state_dict, 'l1.neuron', sigmoid=True)

    ################### test_controller_hidout_file
    # create_connection_from_template('hid2out', torque_state_dict, 'readout.weight')
    create_connection_from_template('hidout', torque_state_dict, 'p_out.synapse.weight')

    ################### test_controller_li_out

    li_out_params = {
        # 'leak': f"{torque_state_dict['p_out.neuron.leak_v'][0].item()}",
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)