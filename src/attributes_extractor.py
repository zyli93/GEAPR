"""User and Business attributes extractor

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        1. 

    TODO:
        1. load data from
        /local2/zyli/irs_fn/data/parse/yelp/preprocess/{user_profile.pkl, business_profile.pkl}
        2. for user:
            2.1 count the coverage of attributes that we are interested in
                - elite (e.g.: [2012, 2013])
                - review_count
                - funny/cool/useful
                - average_stars
                - yelp_since
            2.2 see if there are users that don't have above features
        3. for business:
            3.1 parse diff categories
            3.2 count cat coverage
            3.3 other attributes:
                - longitude ?
                - latitude ?
                - stars
                - review count
                - [isn't there a register time?]
        4. install user/business data features in table


"""

import os
import sys

