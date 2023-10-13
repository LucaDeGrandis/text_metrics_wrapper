from typing import List, Union


def organize_references_in_lists(references: Union[List[str], List[List[str]]]) -> List[List[str]]:
    if isinstance(references[0], str):
        desc_list = [references]
    else:
        max_ref_len = max(len(ref_list) for ref_list in references)
        desc_list = []
        for i in range(max_ref_list):
            desc_list.append([])
            for ref_list in references:
                if len(ref_list) > i:
                    desc_list[i].append(ref_list[i])
                else:
                    desc_list[i].append(None)
    return desc_list
