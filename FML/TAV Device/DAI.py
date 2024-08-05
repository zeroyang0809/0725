import time, random, requests, numpy
import DAN, sys

ServerURL = 'http://140.114.77.93:9999'      #with non-secure connection
#ServerURL = 'https://DomainName' #with SSL connection
Reg_addr = None #if None, Reg_addr=MACaddress

DAN.profile['dm_name']='Vehicle'
DAN.profile['df_list']=['Brake-I', 'Throttle-I', 'V_Gps-I', 'V_Speed-I','Speed_Alert-O']
#DAN.profile['d_name']= 'Assign a Device Name' 

DAN.device_registration_with_retry(ServerURL, Reg_addr)
#DAN.deregister()  #if you want to deregister this device, uncomment this line7拜拜愛俺拜歹ㄉˋ拜拜愛俺拜歹ㄉˋ
#exit()            #if you want to deregister this device, uncomment this line
while True:
    try:
        brake_data = random.getrandbits(1)
        throttle_data = random.getrandbits(1)
        vgps_data_x1 = round(random.uniform(0, 99),2)
        vgps_data_x2 = round(random.uniform(0, 99),2)
        vgps_data_x3 = round(random.uniform(0, 99),2)
        vspeed_data = random.uniform(0, 99)
        
        print(f"{brake_data} {throttle_data} ({vgps_data_x1}, {vgps_data_x2}, {vgps_data_x3}) {vspeed_data}")
        DAN.push ('Brake-I', brake_data) 
        DAN.push ('Throttle-I', throttle_data)
        DAN.push ('V_Gps-I', vgps_data_x1, vgps_data_x2, vgps_data_x3)
        DAN.push ('V_Speed-I', vspeed_data)

        time.sleep(0.3)

    except KeyboardInterrupt:
        try:
            DAN.deregister()    # 試著解除註冊˙
        except Exception as e:
            print("===")
        print("Bye ! -------------", flush=True)
        sys.exit(0)

    except Exception as e:
        print(e)
        print("exception occur")
        if str(e).find('mac_addr not found:') != -1:
            print('Reg_addr is not found. Try to re-register...')
        
