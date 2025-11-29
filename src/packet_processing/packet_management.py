import time

from PySide6.QtWidgets import QListWidget

import src
from src.packet_processing.dictionnaries import *
from src.packet_processing.variables import format_milliseconds
from src.packet_processing.variables import *

# Database integration (non-blocking writes)
try:
    from src.database.data_writer import telemetry_writer
    DATABASE_ENABLED = True
except ImportError:
    DATABASE_ENABLED = False
    print("[PacketManagement] Database module not available - running in memory-only mode")


def update_motion(packet, *args):  # Packet 0
    for i in range(22):
        PLAYERS_LIST[i].worldPositionX = packet.m_car_motion_data[i].m_world_position_x
        PLAYERS_LIST[i].worldPositionZ = packet.m_car_motion_data[i].m_world_position_z


def update_session(packet):  # Packet 1
    session.trackTemperature = packet.m_weather_forecast_samples[0].m_track_temperature
    session.airTemperature = packet.m_weather_forecast_samples[0].m_air_temperature
    session.nbLaps = packet.m_total_laps
    session.time_left = packet.m_session_time_left

    # Database: Create new session when track or session changes
    if session.track != packet.m_track_id or session.Session != packet.m_session_type:
        session.track = packet.m_track_id
        src.packet_processing.variables.REDRAW_MAP = True

        # Queue new session creation
        if DATABASE_ENABLED and hasattr(telemetry_writer, 'create_session'):
            try:
                track_name = tracks_name[packet.m_track_id] if packet.m_track_id in tracks_name else f"Track_{packet.m_track_id}"
                session_type = session_types.get(packet.m_session_type, "Unknown")

                session_id = telemetry_writer.create_session(
                    track_id=packet.m_track_id,
                    track_name=track_name,
                    session_type=session_type,
                    total_laps=packet.m_total_laps,
                    weather=weather_dictionary.get(packet.m_weather, "Unknown"),
                    air_temp=packet.m_weather_forecast_samples[0].m_air_temperature,
                    track_temp=packet.m_weather_forecast_samples[0].m_track_temperature
                )
            except Exception as e:
                pass  # Silent fail - don't break existing functionality

    session.Session = packet.m_session_type
    session.marshalZones = packet.m_marshal_zones  # Array[21]
    session.flag = ""
    for element in session.marshalZones:
        if element.m_zone_flag == 3:
            session.flag = "🟡"
            break
        elif element.m_zone_flag == 1:
            session.flag = "🟢"
    session.marshalZones[0].m_zone_start = session.marshalZones[0].m_zone_start - 1
    session.num_marshal_zones = packet.m_num_marshal_zones
    session.safetyCarStatus = packet.m_safety_car_status
    session.trackLength = packet.m_track_length
    session.clear_slot()
    if packet.m_num_weather_forecast_samples != session.nb_weatherForecastSamples:
        session.nb_weatherForecastSamples = packet.m_num_weather_forecast_samples
    session.weatherList = packet.m_weather_forecast_samples

    # Database: Queue weather sample periodically
    if DATABASE_ENABLED and hasattr(telemetry_writer, 'queue_weather_sample'):
        try:
            # Sample weather every ~30 seconds (approximated by checking session time)
            if hasattr(session, '_last_weather_sample_time'):
                if time.time() - session._last_weather_sample_time > 30:
                    session._last_weather_sample_time = time.time()
                    telemetry_writer.queue_weather_sample({
                        'weather': weather_dictionary.get(packet.m_weather, "Unknown"),
                        'air_temp': packet.m_weather_forecast_samples[0].m_air_temperature,
                        'track_temp': packet.m_weather_forecast_samples[0].m_track_temperature,
                        'rain_percentage': packet.m_rain_percentage if hasattr(packet, 'm_rain_percentage') else 0
                    })
            else:
                session._last_weather_sample_time = time.time()
        except Exception as e:
            pass


def update_lap_data(packet):  # Packet 2
    mega_array = packet.m_lap_data
    for index in range(22):
        element = mega_array[index]
        joueur : Player = PLAYERS_LIST[index]

        joueur.position = element.m_car_position
        joueur.lastLapTime = round(element.m_last_lap_time_in_ms, 3)
        joueur.pit = element.m_pit_status
        joueur.driverStatus = element.m_driver_status
        joueur.penalties = element.m_penalties
        joueur.warnings = element.m_corner_cutting_warnings
        joueur.speed_trap = round(element.m_speedTrapFastestSpeed, 2)
        joueur.currentLapTime = element.m_current_lap_time_in_ms/1000
        joueur.gap_to_car_ahead = element.m_deltaToCarInFrontMSPart/1_000
        joueur.currentLapInvalid = element.m_current_lap_invalid
        joueur.resultStatus = element.m_result_status
        joueur.lapDistance = element.m_lap_distance
        joueur.speedTrapSpeed = element.m_speedTrapFastestSpeed

        if element.m_sector1_time_in_ms == 0 and joueur.currentSectors[0] != 0:  # We start a new lap
            joueur.lastLapSectors = joueur.currentSectors[:]
            joueur.lastLapSectors[2] = joueur.lastLapTime / 1_000 - joueur.lastLapSectors[0] - joueur.lastLapSectors[1]
            joueur.tyre_wear_on_last_lap = ['%.2f' % (float(a)-float(b)) for a,b in zip(joueur.tyre_wear, joueur.tyre_wear_before_last_lap)]
            joueur.tyre_wear_before_last_lap = joueur.tyre_wear[:]

            # Database: Queue completed lap data
            if DATABASE_ENABLED and hasattr(telemetry_writer, 'queue_lap') and joueur.lastLapTime > 0:
                try:
                    lap_number = element.m_current_lap_num - 1  # Last completed lap
                    telemetry_writer.queue_lap(
                        driver_index=index,
                        lap_number=lap_number,
                        lap_data={
                            'driver_name': joueur.name,
                            'lap_time_ms': int(joueur.lastLapTime),
                            'sector1_time_ms': int(joueur.lastLapSectors[0] * 1000) if joueur.lastLapSectors[0] > 0 else None,
                            'sector2_time_ms': int(joueur.lastLapSectors[1] * 1000) if joueur.lastLapSectors[1] > 0 else None,
                            'sector3_time_ms': int(joueur.lastLapSectors[2] * 1000) if joueur.lastLapSectors[2] > 0 else None,
                            'position': joueur.position,
                            'tyre_compound': visual_tyre_compound_dictionary.get(joueur.tyres, "UNKNOWN"),
                            'tyre_age_laps': joueur.tyresAgeLaps,
                            'speed_trap_speed': joueur.speed_trap,
                            'fuel_remaining_laps': joueur.fuelRemainingLaps,
                            'ers_percent': joueur.ERS_pourcentage,
                            'pit_status': joueur.pit,
                            'driver_status': joueur.driverStatus,
                            'result_status': joueur.resultStatus,
                            'penalties': joueur.penalties,
                            'warnings': joueur.warnings,
                            'lap_invalid': joueur.currentLapInvalid
                        }
                    )
                except Exception as e:
                    pass  # Silent fail

        joueur.currentSectors = [element.m_sector1_time_in_ms / 1000, element.m_sector2_time_in_ms / 1000, 0]
        if joueur.bestLapTime > element.m_last_lap_time_in_ms != 0 or joueur.bestLapTime == 0:
            joueur.bestLapTime = element.m_last_lap_time_in_ms
            joueur.bestLapSectors = joueur.lastLapSectors[:]
        if joueur.bestLapTime < session.bestLapTime and element.m_last_lap_time_in_ms != 0 or joueur.bestLapTime == 0:
            session.bestLapTime = joueur.bestLapTime
            session.idxBestLapTime = index
        if element.m_car_position == 1:
            session.currentLap = mega_array[index].m_current_lap_num
            session.tour_precedent = session.currentLap - 1

    players_speed_trap_sorted = sorted(PLAYERS_LIST, key=lambda player: player.speedTrapSpeed, reverse=True)
    for pos, player in enumerate(players_speed_trap_sorted):
        player.speedTrapPosition = pos+1


def update_event(packet, qlist : QListWidget):  # Packet 3
    code = "".join(map(chr, packet.m_event_string_code))
    if code == "STLG" and packet.m_event_details.m_start_lights.m_num_lights >= 2: # Start Lights
        session.formationLapDone = True
        qlist.insertItem(0, f"{packet.m_event_details.m_start_lights.m_num_lights} red lights ")
    elif code == "LGOT" and session.formationLapDone: # Lights out
        qlist.insertItem(0, "Lights out !")
        session.formationLapDone = False
        session.startTime = time.time()
        for joueur in PLAYERS_LIST:  # We reset all the datas (which were from qualifying)
            joueur.reset()
    elif code == "RTMT":  # Retirement
        PLAYERS_LIST[packet.m_event_details.m_retirement.m_vehicle_idx].hasRetired = True
        qlist.insertItem(0, f"{PLAYERS_LIST[packet.m_event_details.m_retirement.m_vehicle_idx].name} retired : " +
                         f"{retirements_dictionnary[packet.m_event_details.m_retirement.m_reason]}")
    elif code == "FTLP":  # Fastest Lap
        qlist.insertItem(0, f"Fastest Lap : {PLAYERS_LIST[packet.m_event_details.m_fastest_lap.m_vehicle_idx].name} - "
                            f"{format_milliseconds(packet.m_event_details.m_fastest_lap.m_lap_time*1000)}")
    elif code == "DRSD":  # DRS Disabled
        qlist.insertItem(0, f"DRS Disabled : {drs_disabled_reasons[packet.m_event_details.m_drs_disabled.m_reason]}")
    elif code == "DRSE":  # DRS Enabled
        qlist.insertItem(0, "DRS Enabled")
    elif code == "CHQF":
        qlist.insertItem(0, "Chequered Flag")


def update_participants(packet):  # Packet 4
    if session.nb_players != packet.m_num_active_cars:
        src.packet_processing.variables.REDRAW_MAP = True
        session.nb_players = packet.m_num_active_cars

    for index in range(22):
        element = packet.m_participants[index]
        joueur = PLAYERS_LIST[index]
        joueur.raceNumber = element.m_race_number
        joueur.teamId = element.m_team_id
        joueur.aiControlled = element.m_ai_controlled
        joueur.yourTelemetry = element.m_your_telemetry
        if joueur.networkId != element.m_network_id:
            joueur.networkId = element.m_network_id
            src.packet_processing.variables.REDRAW_MAP = True
        try:
            joueur.name = element.m_name.decode("utf-8")
        except:
            joueur.name = element.m_name

        if joueur.name in ['Pilote', 'Driver']: # More translations appreciated
            joueur.name = teams_name_dictionary[joueur.teamId] + "#" + str(joueur.raceNumber)

def update_car_setups(packet): # Packet 5
    array = packet.m_car_setups
    for index in range(22):
        PLAYERS_LIST[index].setup_array = array[index]

def update_car_telemetry(packet):  # Packet 6
    for index in range(22):
        element = packet.m_car_telemetry_data[index]
        joueur = PLAYERS_LIST[index]
        joueur.drs = element.m_drs
        joueur.tyres_temp_inner = element.m_tyres_inner_temperature
        joueur.tyres_temp_surface = element.m_tyres_surface_temperature
        joueur.speed = element.m_speed
        if joueur.speed >= 200 and not joueur.S200_reached:
            print(f"{joueur.position} {joueur.name}  = {time.time() - session.startTime}")
            joueur.S200_reached = True

    # Database: Queue telemetry snapshots for player car only (to reduce data volume)
    # Full 22-car telemetry at 60 FPS would be ~1.3GB/hour, so we sample player car
    if DATABASE_ENABLED and hasattr(telemetry_writer, 'queue_telemetry'):
        try:
            # Find player index (human player has yourTelemetry == 1)
            for index in range(22):
                if PLAYERS_LIST[index].yourTelemetry == 1:
                    element = packet.m_car_telemetry_data[index]
                    joueur = PLAYERS_LIST[index]

                    telemetry_writer.queue_telemetry(
                        driver_index=index,
                        lap_number=session.currentLap,
                        telemetry_data={
                            'speed_kph': element.m_speed,
                            'throttle': element.m_throttle,
                            'brake': element.m_brake,
                            'steering': element.m_steer,
                            'gear': element.m_gear,
                            'engine_rpm': element.m_engine_rpm,
                            'drs_active': element.m_drs == 1,
                            'tyre_surface_temp_fl': element.m_tyres_surface_temperature[2],  # RL
                            'tyre_surface_temp_fr': element.m_tyres_surface_temperature[3],  # RR
                            'tyre_surface_temp_rl': element.m_tyres_surface_temperature[0],  # FL
                            'tyre_surface_temp_rr': element.m_tyres_surface_temperature[1],  # FR
                            'tyre_inner_temp_fl': element.m_tyres_inner_temperature[2],
                            'tyre_inner_temp_fr': element.m_tyres_inner_temperature[3],
                            'tyre_inner_temp_rl': element.m_tyres_inner_temperature[0],
                            'tyre_inner_temp_rr': element.m_tyres_inner_temperature[1]
                        }
                    )
                    break  # Only player car
        except Exception as e:
            pass  # Silent fail

def update_car_status(packet):  # Packet 7
    for index in range(22):
        element = packet.m_car_status_data[index]
        joueur = PLAYERS_LIST[index]
        joueur.fuelMix = element.m_fuel_mix
        joueur.fuelRemainingLaps = element.m_fuel_remaining_laps
        joueur.tyresAgeLaps = element.m_tyres_age_laps

        # Database: Queue tyre data when compound changes (pit stop)
        if joueur.tyres != element.m_visual_tyre_compound:
            if DATABASE_ENABLED and hasattr(telemetry_writer, 'queue_tyre_data'):
                try:
                    telemetry_writer.queue_tyre_data(
                        driver_index=index,
                        lap_number=session.currentLap,
                        tyre_data={
                            'compound': visual_tyre_compound_dictionary.get(element.m_visual_tyre_compound, "UNKNOWN"),
                            'age_laps': element.m_tyres_age_laps,
                            'wear_fl': float(joueur.tyre_wear[2]) if len(joueur.tyre_wear) > 2 else 0,
                            'wear_fr': float(joueur.tyre_wear[3]) if len(joueur.tyre_wear) > 3 else 0,
                            'wear_rl': float(joueur.tyre_wear[0]) if len(joueur.tyre_wear) > 0 else 0,
                            'wear_rr': float(joueur.tyre_wear[1]) if len(joueur.tyre_wear) > 1 else 0,
                            'surface_temp_fl': joueur.tyres_temp_surface[2] if len(joueur.tyres_temp_surface) > 2 else 0,
                            'surface_temp_fr': joueur.tyres_temp_surface[3] if len(joueur.tyres_temp_surface) > 3 else 0,
                            'surface_temp_rl': joueur.tyres_temp_surface[0] if len(joueur.tyres_temp_surface) > 0 else 0,
                            'surface_temp_rr': joueur.tyres_temp_surface[1] if len(joueur.tyres_temp_surface) > 1 else 0,
                            'inner_temp_fl': joueur.tyres_temp_inner[2] if len(joueur.tyres_temp_inner) > 2 else 0,
                            'inner_temp_fr': joueur.tyres_temp_inner[3] if len(joueur.tyres_temp_inner) > 3 else 0,
                            'inner_temp_rl': joueur.tyres_temp_inner[0] if len(joueur.tyres_temp_inner) > 0 else 0,
                            'inner_temp_rr': joueur.tyres_temp_inner[1] if len(joueur.tyres_temp_inner) > 1 else 0
                        }
                    )
                except Exception as e:
                    pass  # Silent fail

            joueur.tyres = element.m_visual_tyre_compound

        joueur.ERS_mode = element.m_ers_deploy_mode
        joueur.ERS_pourcentage = round(element.m_ers_store_energy / 40_000)
        joueur.DRS_allowed = element.m_drs_allowed
        joueur.DRS_activation_distance = element.m_drs_activation_distance

def update_car_damage(packet):  # Packet 10
    for index in range(22):
        element = packet.m_car_damage_data[index]
        joueur = PLAYERS_LIST[index]

        # Database: Queue damage event when significant damage occurs
        if DATABASE_ENABLED and hasattr(telemetry_writer, 'queue_damage_event'):
            try:
                # Calculate total damage
                total_damage = (element.m_front_left_wing_damage + element.m_front_right_wing_damage +
                               element.m_rear_wing_damage + element.m_floor_damage +
                               element.m_diffuser_damage + element.m_sidepod_damage)

                # Only queue if there's significant damage (> 5% on any component)
                if (element.m_front_left_wing_damage > 5 or element.m_front_right_wing_damage > 5 or
                    element.m_rear_wing_damage > 5 or element.m_floor_damage > 5 or
                    element.m_diffuser_damage > 5 or element.m_sidepod_damage > 5):

                    # Track last recorded damage to avoid duplicates
                    if not hasattr(joueur, '_last_damage_total'):
                        joueur._last_damage_total = 0

                    # Only queue if damage increased by at least 2%
                    if total_damage - joueur._last_damage_total >= 2:
                        telemetry_writer.queue_damage_event(
                            driver_index=index,
                            lap_number=session.currentLap,
                            damage_data={
                                'front_left_wing_damage': element.m_front_left_wing_damage,
                                'front_right_wing_damage': element.m_front_right_wing_damage,
                                'rear_wing_damage': element.m_rear_wing_damage,
                                'floor_damage': element.m_floor_damage,
                                'diffuser_damage': element.m_diffuser_damage,
                                'sidepod_damage': element.m_sidepod_damage,
                                'front_left_wing_delta': element.m_front_left_wing_damage - (joueur.frontLeftWingDamage if hasattr(joueur, 'frontLeftWingDamage') else 0),
                                'front_right_wing_delta': element.m_front_right_wing_damage - (joueur.frontRightWingDamage if hasattr(joueur, 'frontRightWingDamage') else 0),
                                'rear_wing_delta': element.m_rear_wing_damage - (joueur.rearWingDamage if hasattr(joueur, 'rearWingDamage') else 0),
                                'damage_type': 'COLLISION' if total_damage - joueur._last_damage_total > 10 else 'WEAR',
                                'severity': 'HIGH' if total_damage > 30 else 'MEDIUM' if total_damage > 10 else 'LOW'
                            }
                        )
                        joueur._last_damage_total = total_damage
            except Exception as e:
                pass  # Silent fail

        joueur.tyre_wear = ['%.2f'%tyre for tyre in element.m_tyres_wear]
        joueur.tyre_blisters = ['%.2f'%tyre for tyre in element.m_tyre_blisters]
        joueur.frontLeftWingDamage = element.m_front_left_wing_damage
        joueur.frontRightWingDamage = element.m_front_right_wing_damage
        joueur.rearWingDamage = element.m_rear_wing_damage
        joueur.floorDamage = element.m_floor_damage
        joueur.diffuserDamage = element.m_diffuser_damage
        joueur.sidepodDamage = element.m_sidepod_damage

def update_motion_extended(packet):  # Packet 13
    #print(list(packet.get_value("m_wheelVertForce")))
    #print(packet.get_value("m_front_aero_height"), packet.get_value("m_rear_aero_height"))
    return

    print()

def nothing(packet):# Packet 8, 9, 11, 12, 13
    pass










