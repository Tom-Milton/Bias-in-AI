import json

def create_diag_dict():
    """Creates mapping dict for diag_x using
    https://www.hindawi.com/journals/bmri/2014/781670/tab2/"""

    diag_dict = {i: 'Other' for i in range(1, 1000)}

    diag_dict.update({'E' + str(i).zfill(3): 'Other' for i in range(1, 1000)})
    diag_dict.update({'V' + str(i).zfill(2): 'Other' for i in range(1, 100)})
    diag_dict.update({365.44: 'Other'})

    diag_dict.update({i: 'Circulatory' for i in range(390, 460)})
    diag_dict.update({785: 'Circulatory'})
    diag_dict.update({i: 'Respiratory' for i in range(460, 520)})
    diag_dict.update({786: 'Respiratory'})
    diag_dict.update({i: 'Digestive' for i in range(520, 580)})
    diag_dict.update({787: 'Digestive'})
    diag_dict.update({'250.' + str(i): 'Diabetes' for i in range(1, 100)})
    diag_dict.update({'250.' + str(i).zfill(2): 'Diabetes' for i in range(1, 10)})
    diag_dict.update({i: 'Injury' for i in range(800, 1000)})
    diag_dict.update({i: 'Musculoskeletal' for i in range(710, 740)})
    diag_dict.update({i: 'Genitourinary' for i in range(580, 630)})
    diag_dict.update({788: 'Genitourinary'})
    diag_dict.update({i: 'Neoplasms' for i in range(140, 240)})

    return diag_dict


def create_admission_type_dict():
    """Create dict for admission_type_id using 'IDs_mapping.csv'"""

    admission_type_dict = {
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective',
        4: 'Newborn',
        5: 'Not available',
        6: 'NULL',
        7: 'Trauma center',
        8: 'Not mapped'
    }

    return admission_type_dict

def create_admission_source_dict():
    """Creates dict for admission_source_id using
    https://www.hindawi.com/journals/bmri/2014/781670/tab3/"""

    admission_source_dict = {i: 'Other admission' for i in range(1, 27)}

    admission_source_dict.update({7: 'Admitted from emergency room'})
    admission_source_dict.update({1: 'Admitted because of physician/clinic referral',
                                  2: 'Admitted because of physician/clinic referral'})

    return admission_source_dict


def create_discharge_disposition_dict():
    """Creates dict for discharge_disposition_id using
    https://www.hindawi.com/journals/bmri/2014/781670/tab3/"""

    discharge_dict = {i: 'Other disposition' for i in range(1, 30)}

    discharge_dict.update({1: 'Discharged to home'})

    return discharge_dict


def main():
    # Retrieves dictionaries
    diag_dict = create_diag_dict()
    admission_type_dict = create_admission_type_dict()
    admission_source_dict = create_admission_source_dict()
    discharge_dict = create_discharge_disposition_dict()

    # Dumps dictionaries for use in main implementation code
    with open('dict_mappings.json', 'w+') as f:
        json.dump([diag_dict, admission_type_dict, admission_source_dict, discharge_dict], f)


main()
