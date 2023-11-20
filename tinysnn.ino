#include <Arduino.h>

#include "msgs/controlInput.h"
#include "msgs/controlOutput.h"

extern "C"{
    #include "NetworkController.h"
    #include "functional.h"
    #include "param/test_controller_conf.h"
}

// -------------------------- COMMUNICATION DEFINED VARIABLES-----------------------------
#define COMMUNICATION_SERIAL Serial2
#define COMMUNICATION_SERIAL_BAUD 460800

byte START_BYTE_SERIAL_CF=0x9A;
elapsedMicros last_time_write_to_cf = 0;
int ack_comm_TX_cf = 0; 

struct serial_control_in myserial_control_in;
uint8_t serial_cf_msg_buf_in[ 2*sizeof(struct serial_control_in) ] = {0};
uint16_t serial_cf_buf_in_cnt = 0;
int serial_cf_received_packets = 0;
int serial_cf_missed_packets_in = 0;

volatile struct serial_control_out myserial_control_out;
volatile float extra_data_out[255]__attribute__((aligned));

bool sending;
bool receiving = true;

// -------------------------- DEBUG DEFINED VARIABLES-----------------------------
#define DEBUG_serial Serial
#define DEBUG_serial_baud 115200

// -------------------------- CONTROL DEFINED VARIABLES-------------------------------
elapsedMicros timer_count_main = 0;
NetworkController controller;
float roll_integ = 0.0f;
float pitch_integ = 0.0f;

// -------------------------- INPUT DEFINED VARIABLES-----------------------------
float gyro_x = 0.0f;
float gyro_y = 0.0f;
float gyro_z = 0.0f;
float acc_x = 0.0f;
float acc_y = 0.0f;
float acc_z = 0.0f;
float roll_target = 0.0f;
float pitch_target = 0.0f;
float inputs[8] = {gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, roll_target, pitch_target};

///////////////////////////////////////////////USER DEFINED FCN///////////////////
void serialParseMessageIn(void)
{
  //Copy received buffer to structure
  memmove(&myserial_control_in,&serial_cf_msg_buf_in[1],sizeof(struct serial_control_in)-1);
  // DEBUG_serial.write("Correct message received and storing\n");
//   DEBUG_serial.write("Stored pitch is %i\n", myserial_control_in.pitch);
}

void setInputMessage(void)
{
    inputs[0] = myserial_control_in.roll_gyro * 0.01;
    inputs[1] = myserial_control_in.pitch_gyro * 0.01;
    inputs[2] = myserial_control_in.yaw_gyro * 0.01;
    inputs[3] = myserial_control_in.x_acc;
    inputs[4] = myserial_control_in.y_acc;
    inputs[5] = myserial_control_in.z_acc * 0.3;
    inputs[6] = myserial_control_in.roll * 0.03;
    inputs[7] = myserial_control_in.pitch * 0.03;
    // inputs = [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, roll_target, pitch_target];
    // DEBUG_serial.printf("%f, %f, %f, %f, %f, %f, %f, %f\n", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
    set_network_input(&controller, inputs);
}

void setOutputMessage(void)
{
    myserial_control_out.torque_x = controller.out[0] * 20000;
    myserial_control_out.torque_y = controller.out[1] * 20000;
    // myserial_control_out.x_integ = roll_integ * 20;
    // myserial_control_out.y_integ = pitch_integ * 20;
    myserial_control_out.x_integ = controller.integ_out[0] * 20000;
    myserial_control_out.y_integ = controller.integ_out[1] * 20000;
}

void sendCrazyflie(void)
{
    //SENDING PACKET

    //Calculate checksum for outbound packet: 
    uint8_t *buf_send = (uint8_t *)&myserial_control_out;
    myserial_control_out.checksum_out = 0;
    for(uint16_t i = 0; i < sizeof(struct serial_control_out) - 1; i++){
        myserial_control_out.checksum_out += buf_send [i];
    }
    
    //Send out packet to buffer:
    noInterrupts();
    COMMUNICATION_SERIAL.write(START_BYTE_SERIAL_CF);
    COMMUNICATION_SERIAL.write(buf_send,sizeof(struct serial_control_out));
    interrupts();
    
    last_time_write_to_cf = 0;

    sending = false;
    receiving = true;
}


void receiveCrazyflie(void)
{
  //RECEIVING PACKET
  //Collect packets on the buffer if available:
    while(COMMUNICATION_SERIAL.available()) {
        // DEBUG_serial.write("trying to read...\n");
        uint8_t serial_cf_byte_in;
        serial_cf_byte_in = COMMUNICATION_SERIAL.read();
        if ((serial_cf_byte_in == START_BYTE_SERIAL_CF) || (serial_cf_buf_in_cnt > 0)) {
            serial_cf_msg_buf_in[serial_cf_buf_in_cnt] = serial_cf_byte_in;
            serial_cf_buf_in_cnt++;
        }
        if (serial_cf_buf_in_cnt > sizeof(struct serial_control_in)  ) {
            serial_cf_buf_in_cnt = 0;
            uint8_t checksum_in_local = 0;
            for(uint16_t i = 1; i < sizeof(struct serial_control_in) ; i++){
                checksum_in_local += serial_cf_msg_buf_in[i];
            }
            if(checksum_in_local == serial_cf_msg_buf_in[sizeof(struct serial_control_in)]){
                serialParseMessageIn();
                serial_cf_received_packets++;
                
            }
            else {
                serial_cf_missed_packets_in++;           
                DEBUG_serial.write("Incorrect message\n");
            }
            receiving = false;
            sending = true;
        }
    }
}

void setup(void) 
{
    //////////////////SETUP DEBUGGING USB
    DEBUG_serial.begin(DEBUG_serial_baud);


    //////////////////Initialize controller network
    DEBUG_serial.write("Build network\n");
    controller = build_network(8, 80, 80, 4, 80, 2);
    DEBUG_serial.write("Init network\n");
    init_network(&controller);

    // Load network parameters from header file and reset
    DEBUG_serial.write("Loading network\n");
    load_network_from_header(&controller, &conf);
    DEBUG_serial.write("Resetting\n");
    reset_network(&controller);

    //////////////////SETUP CONNECTION WITH CRAZYFLIE
    COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD);
    DEBUG_serial.write("Finished setup\n");
}

///////////////////////////////////////////////////////////LOOP///////////////////
void loop(void) 
{
    if (receiving) {
        // DEBUG_serial.write("receiving...");
        receiveCrazyflie();
    } else if (sending) {
        // Timer for debugging
        if (timer_count_main > 1000000) {
          DEBUG_serial.printf("Received %i packets over last second\n", serial_cf_received_packets);
          serial_cf_received_packets = 0;
          timer_count_main = 0;
        }
        // Set input to network from CF
        // DEBUG_serial.write("Setting input message\n");
        setInputMessage();

        // Reset network if thrust command is zero
        // TODO: Find better solution, otherwise network might be reset mid flight
        if (myserial_control_in.thrust == 0.0f) {
          reset_network(&controller);
          roll_integ = 0.0f;
          pitch_integ = 0.0f;
        }

        // Forward network
        forward_network(&controller);
        roll_integ += controller.out[0] - 5 * controller.out[2];
        pitch_integ += controller.out[1] + 5 * controller.out[3];

        // Store output message to be sent back to CF
        setOutputMessage();

        // Send message via UART to CF
        sendCrazyflie();
    }
}