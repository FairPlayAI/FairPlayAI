import json
import random
import os
import pandas as pd
import numpy as np

def dataframe(train_path: str):
    """
    Args:
        train_path: Placeholder for the train folder containing train actions subset. (.../data/train/)
    """
    column_names = ['action_id', 'clip_path', 'Action_class', 'Offence', 'Severity', 'Handball', 'Handball_offence',
                    'Contact', 'BodyPart', 'Upper_body_part', 'Multiple_fouls', 'Try_to_play', 'Touch_ball', 'replay_speed']
    data = pd.DataFrame(columns=column_names)

    with open(f"{train_path}/annotations.json", 'r') as file:
        json_data = json.load(file)

    for i in range(2916):
        if json_data['Actions'][f'{i}']['Action class'] == '' or json_data['Actions'][f'{i}']['Action class'] == "Dont know":
            continue

        if (json_data['Actions'][f'{i}']['Offence'] == '' or json_data['Actions'][f'{i}']['Offence'] == 'Between') and json_data['Actions'][f'{i}']['Action class'] != 'Dive':
            continue

        if (json_data['Actions'][f'{i}']['Severity'] == '' or json_data['Actions'][f'{i}']['Severity'] == '2.0' or json_data['Actions'][f'{i}']['Severity'] == '4.0') and json_data['Actions'][f'{i}']['Action class'] != 'Dive' and json_data['Actions'][f'{i}']['Offence'] != 'No Offence':
            continue

        clips = json_data['Actions'][f'{i}']['Clips']
        if json_data['Actions'][f'{i}']['Offence'] == '' or json_data['Actions'][f'{i}']['Offence'] == 'Between':
            offence = 'Offence'
        else:
            offence = json_data['Actions'][f'{i}']['Offence']

        contact = json_data['Actions'][f'{i}']['Contact']
        body_part = json_data['Actions'][f'{i}']['Bodypart']
        upper_body_part = json_data['Actions'][f'{i}']['Upper body part']
        multiple_fouls = json_data['Actions'][f'{i}']['Multiple fouls']
        try_to_play = json_data['Actions'][f'{i}']['Try to play']
        touch_ball = json_data['Actions'][f'{i}']['Touch ball']
        action_class = json_data['Actions'][f'{i}']['Action class']
        if json_data['Actions'][f'{i}']['Severity'] == '' or json_data['Actions'][f'{i}']['Severity'] == '2.0' or json_data['Actions'][f'{i}']['Severity'] == '4.0':
            severity = '1.0'
        else:
            severity = json_data['Actions'][f'{i}']['Severity']
        handball = json_data['Actions'][f'{i}']['Handball']
        handball_offence = json_data['Actions'][f'{i}']['Handball offence']

        prev = []
        cnt = 0
        cont = True
        if len(clips) == 2:
            for j in range(len(clips)):
                path_to_be_in_df = os.path.join(
                    train_path, f'action_{i}/clip_{j}.mp4')
                replay_speed = json_data['Actions'][f'{i}']['Clips'][j]['Replay speed']
                data.loc[len(data)] = [i,
                                       path_to_be_in_df, action_class, offence, severity, handball, handball_offence,
                                       contact, body_part, upper_body_part, multiple_fouls, try_to_play, touch_ball,
                                       replay_speed
                                       ]
        else:
            while cont:
                aux = random.randint(0, len(clips)-1)
                if aux not in prev:
                    prev.append(aux)
                    path_to_be_in_df = os.path.join(
                        train_path, f'action_{i}/clip_{aux}.mp4')
                    replay_speed = json_data['Actions'][f'{i}']['Clips'][aux]['Replay speed']
                    data.loc[len(data)] = [i,
                                           path_to_be_in_df, action_class, offence, severity, handball, handball_offence,
                                           contact, body_part, upper_body_part, multiple_fouls, try_to_play, touch_ball,
                                           replay_speed
                                           ]
                    cnt = cnt + 1
                if cnt == 2:
                    cont = False
    data['Offence'] = data['Offence'].map(lambda x: 1.0 if x == 'Offence' else 0.0)
    data['card'] = data['Severity'].astype(np.float64) * data['Offence']
    return data
