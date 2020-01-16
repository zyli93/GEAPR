# Dataset description

1. `processed_business_user_profile.csv`: 
    * location: `/local2/zyli/irs_fn/data/parse/yelp/citycluster/lv` (or `tor`, `phx`)
    * content: processed user information, the easiest way to load users' information.
2. `train_pos.csv`
    * location: `/local2/zyli/irs_fn/data/parse/yelp/train_test/lv`
    * content: positive training data (90%) of overall data.
    * format: `[uid] [bid]` for each row.
3. `test_instances`
    * location: `/local2/zyli/irs_fn/data/parse/yelp/train_test/lv`
    * content: test user and their associated positive examples (10%)
    * format: `{uid1:[bid1, bid2, ...], uid2:[...], ...}`

