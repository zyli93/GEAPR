
def generate_data(city, ratio, negative_sample_ratio):
    """
    Create training set and test set

    Arg:
        city: the city to work on, (str)
        ratio: train/test ratio, (tuple)

    Store:
        train.csv - training data csv
        test.csv - testing data.csv
        valid.csv - validation data csv
    """
    ub = pd.read_csv(CITY_DIR + CITY_NAME_ABBR[city] + "/user_business_interaction.csv")

    users = ub.user.tolist()
    businesses = ub.business.tolist()

    print("\t[--gen_data] {}: Zipping positive samples ...".format(city))
    pos_samples = set(zip(users, businesses))
    pos_count = ub.shape[1]

    neg_samples = []

    # Sample positive samples and negative samples
    # TODO: may need to think of better sampling algorithms
    while len(neg_samples) < pos_count * neg_ratio:
        sample_u = np.random.choice(users)
        sample_b = np.random.choice(businesses)
        if (sample_u, sample_b) not in pos_samples:
            neg_samples.append((sample_u, sample_b))

    neg_samples = list(zip(*neg_samples))
    df_neg = pd.DataFrame({"user": neg_samples[0], "business": neg_samples[1], "label": 0})

    df_pos = ub[['user', 'business']]
    df_pos = df_pos.assign(label=1)  # use df.assign as a better way to append new columns

    # ratio: Train, Test, Validation
    df_data = pd.concat([df_neg, df_pos], axis=0, ignore_index=True, sort=False)

    print("\t\tRatio: {}:{}:{};".format(*ratio), end=" ")
    print("(Trn+Val:Tst): {}; (Trn:Tst): {}"
          .format(ratio[1]/sum(ratio), ratio[2]/(ratio[0]+ratio[2])))

    train_df, test_df = train_test_split(df_data, random_state=723, test_size=(ratio[1]/sum(ratio)))
    train_df, valid_df = train_test_split(train_df, random_state=723,
        test_size=(ratio[2]/(ratio[0]+ratio[2])))

    city_interaction_dir = INTERACTION_DIR + CITY_NAME_ABBR[city] + "/"
    make_dir(INTERACTION_DIR)
    make_dir(city_interaction_dir)
    train_df.to_csv(city_interaction_dir + "train.csv", index=False)
    test_df.to_csv(city_interaction_dir + "test.csv", index=False)
    valid_df.to_csv(city_interaction_dir + "valid.csv", index=False)

    print("\t[--gen_data] {}: Finished! Data generated at {}".format(city, city_interaction_dir))

