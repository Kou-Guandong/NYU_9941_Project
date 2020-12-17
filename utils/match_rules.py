import yaml


def get_rules(rule_name='job_type'):
    with open("manual_rules/%s_rules.yml" % rule_name, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


job_type_rules = get_rules('job_type')
job_exp_rules = get_rules('job_exp')


def match_rule_with_title(title, rules):
    for item in rules:
        for keyword in rules[item]['title']:
            if keyword in title:
                return item


def match_job_type_rules(title, desc=''):
    return match_rule_with_title(title, job_type_rules)


def match_job_exp_rules(title, desc=''):
    return match_rule_with_title(title, job_exp_rules)
