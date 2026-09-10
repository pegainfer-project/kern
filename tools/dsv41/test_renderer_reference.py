"""Generate CPU renderer golden cases from the released V4.1 encoding oracle."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('encoding',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('v41_encoding',a.encoding/'encoding.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    helpers=[n for n in ast.parse((a.encoding/'test_encoding.py').read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name in ('make_tool','make_tool_call_messages')]
    scope={};exec(compile(ast.Module(body=helpers,type_ignores=[]),'official_encoding_test_helpers','exec'),scope)
    cases=[]
    def add(name,messages,thinking=True,effort=None,drop=True):
        case={'name':name,'messages':copy.deepcopy(messages),'thinking':thinking,'effort':effort,'drop_thinking':drop}
        try:case['expected']=ref.encode_messages(messages,thinking_mode='thinking' if thinking else 'chat',reasoning_effort=effort,drop_thinking=drop)
        except AssertionError:case['error']=True
        cases.append(case)
    for effort in (None,'low','high','xhigh','max',1,42,100,-1,0,101,'medium',True,False,1.5):
        add('effort-'+json.dumps(effort),[{'role':'user','content':'question'}],effort=effort)
    add('chat',[{'role':'user','content':'hello'}],False,'max')
    add('initial-system',[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'hello'}],False)
    history=[{'role':'system','content':'sys'},{'role':'user','content':'q1'},{'role':'assistant','content':'a1','reasoning_content':'r1'},{'role':'system','content':'mid sys'}]
    add('mid-system',history,effort=88)
    history[-1]={'role':'user','content':'q2'}
    add('drop-reasoning',history);add('keep-reasoning',history,drop=False)
    tool=scope['make_tool']();messages=scope['make_tool_call_messages']()
    add('assistant-spaced-tools',messages)
    for thinking in (False,True):
        add('tool-schemas-'+str(thinking),[{'role':'system','content':'system','tools':[tool]},{'role':'user','content':'question'}],thinking)
    add('consecutive-user',[{'role':'user','content':'first'},{'role':'user','content':'second'}],False)
    messages[1]['tool_calls'][0]['id']='a'
    messages[1]['tool_calls'].append({'id':'b','type':'function','function':{'name':'lookup','arguments':'"{\\"query\\":\\"double\\"}"'}})
    messages += [{'role':'tool','tool_call_id':'b','content':'second'},{'role':'user','content':'note'},{'role':'tool','tool_call_id':'a','content':'first'}]
    add('mixed-tool-user-sorted',messages)
    messages=copy.deepcopy(messages[:2]);messages[1]['tool_calls'][0]['function']['arguments']='not json'
    add('invalid-json-tool-argument-fallback',messages)
    for index in (1,2):
        case=ref.load_cases(str(a.encoding/'tests'/f'test_input_{index}.json'))[0]
        assert not case.get('context')
        add('released-golden-'+str(index),case['messages'],case.get('thinking_mode')=='thinking',case.get('reasoning_effort'))
        assert cases[-1]['expected']==(a.encoding/'tests'/f'test_output_{index}.txt').read_text()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(cases)} official reference cases written')


if __name__=='__main__':main()
