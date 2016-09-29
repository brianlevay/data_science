# This module returns the most frequent domains for HackerNews articles

import read

def get_domain(url):
    try:
        parts = url.split('.')
        num = len(parts)
        domain = ".".join([parts[num-2],parts[num-1]])
        return domain
    except:
        return url


def freq_domains():
    # read the data
    data = read.load_data()
    
    # returns a series containing only the domain parts of the urls
    domains = data["url"].apply(get_domain)
    
    # get the counts of each unique url (domain)
    domain_cts = domains.value_counts(ascending=False)
    
    # print results
    counter = 1
    for name, row in domain_cts.items():
        if counter == 100:
            break
        else:
            print("{0}: {1}".format(name,row))
            counter += 1
    
    return domain_cts


if __name__ == "__main__":
    freq_domains()