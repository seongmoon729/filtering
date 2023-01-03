import sys


def write_replaced_config_file(base_src_path, target_src_path, dst_path):
    c1 = read_config_file(base_src_path)
    c2 = read_config_file(target_src_path)
    for k in set(c1.keys()) & set(c2.keys()):
        c1[k] = c2[k]
    # Due to the issue.
    c1['GOPSize'] = 2
    c1['SignHideFlag'] = 0
    with open(dst_path, 'w') as f:
        for k, v in c1.items():
            f.write(f'{k:35} : {v}\n')


def read_config_file(path):
    configs = open(path, 'r').readlines()
    configs = filter(lambda x: '==' not in x and x != '\n' and x[0] != '#', configs)
    configs = map(lambda x: x.split('#')[0], configs)
    keys, values = zip(*map(lambda x: x.split(':'), configs))
    keys = map(lambda x: x.strip(), keys)
    values = map(lambda x: x.strip(), values)
    configs = {k: v for k, v in zip(keys, values)}
    return configs 


if __name__ == '__main__':
    base_src_path, target_src_path, dst_path = sys.argv[1:]
    write_replaced_config_file(base_src_path, target_src_path, dst_path)