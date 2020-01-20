
def extract_business_attr(city):
    """extract business attributes

    Args:
        city - the city to profess
    Print-outs:
        df_nonzero - non zero number ratios of each attribute
    Store:
        bus_cat_dicts - business category
    Return:
        df_nonzero - as above.
    """

    print("\t[business] loading business interaction {}...".format(city))
    with open(INPUT_DIR + "{}/city_business_profile.pkl".format(city), "rb") as fin:
        business_profile = pickle.load(fin)

    business_data_csv = []
    business_data_pkl = {}

    bus_cat_dicts = {EMPTY_CATS: 0}

    # process users, NOTE: user new index starts with 1
    for bid, prof_dict in business_profile.items():
        # --- create feature area ---
        tmp_entry = dict()
        categories = prof_dict.get("categories")
        if not categories:  # one outlier without categories (recorded as `None`)
            tmp_entry['category_indices'] = [EMPTY_CATS]
        else:
            categories = categories.strip().split(", ")
            tmp_entry['category_indices'] = []
            for cat in categories:
                if cat not in bus_cat_dicts:
                    bus_cat_dicts[cat] = len(bus_cat_dicts)
                tmp_entry['category_indices'].append(bus_cat_dicts[cat])
        tmp_entry['review_count'] = prof_dict.get('review_count', CNT_DFL)  # review_count

        tmp_entry['stars'] = prof_dict.get('stars', STAR_DFL)
        tmp_entry['name'] = prof_dict.get('name', NAME_DFL)  # confirmed that all bus have names
        tmp_entry['longitude'] = prof_dict.get('longitude')
        tmp_entry['latitude'] = prof_dict.get('latitude')
        # --- end create feature area ---

        business_data_csv.append(tmp_entry)
        business_data_pkl[bid] = tmp_entry

    # create dataframe
    df_business_profile = pd.DataFrame(business_data_csv)

    # non-zero
    df_nonzero = df_business_profile.fillna(0).astype(bool).sum(axis=0)
    df_nonzero = df_nonzero / len(df_business_profile)
    print("\t[business] non-zero terms in `df_business_profile`")
    print(df_nonzero)

    print("\t[business] saving dataframe to {}".format(OUTPUT_DIR))
    df_business_profile.to_csv(
        OUTPUT_DIR+"{}/processed_city_business_profile.csv".format(city), index=False)
    dump_pkl(OUTPUT_DIR+"{}/processed_city_business_profile.pkl".format(city), business_data_pkl)

    print("\t[business] saving categories dictionary")
    dump_pkl(OUTPUT_DIR+"{}/bus_cat_idx_dict.pkl".format(city), bus_cat_dicts)

    return df_nonzero
