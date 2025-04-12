from lm_eval.tasks import TaskManager
import json
import os 

task_manager = TaskManager()
# To add custom dataset
# 1. Add config fiile to > lm_eval/tasks/{your_taskname}/{task_config.yaml} 
# 2. TaskManager(include_path='lm_eval/tasks/{your_taskname}/')

# TODO: @suhyun check the task list to train
task_list = ["hellaswag", "gsm8k", "openbookqa", "arc_easy", "arc_challenge", "piqa", "winogrande", "commonsense_qa"]

output_dir = "./data/train"
# Output will be 
# Sample : {output_dir}/samples/{task_name}-sample1.jsonl
# Dataset : {output_dir}/{task_name}.jsonl 

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/samples', exist_ok=True)

tasks = task_manager.load_task_or_group(task_list)
print(f'task_list   : {task_list}')

# To get the eval dataset set this as False
extract_train_data = True 

for tname, tobj in tasks.items():
    tobj.build_all_requests(training=extract_train_data)
    instances = tobj.instances
    
    print(f'tname : {tname} | len : {len(instances)} ==========================') 
    
    with open(f'{output_dir}/samples/{tname}-sample1.jsonl', 'w') as f:
        raw_doc = instances[0].doc 
        prompt = instances[0].args[0]
        target = tobj.doc_to_target(instances[0].doc) 
        if tobj.config.doc_to_choice and not isinstance(target, str):
            target = tobj.doc_to_choice(instances[0].doc)[target]
            
        f.write(json.dumps({"prompt": prompt, "target":target, "raw":raw_doc}, indent=4))
        print(json.dumps({"prompt": prompt, "target":target, "raw":raw_doc}, indent=4))

    
    with open(f'{output_dir}/{tname}.jsonl', 'w') as f:
        for i in instances :
            raw_doc = i.doc 
            prompt = i.args[0]
            target = tobj.doc_to_target(i.doc)
            if tobj.config.doc_to_choice and not isinstance(target, str):
                target = tobj.doc_to_choice(i.doc)[target]
            f.write(json.dumps({"prompt": prompt, "target":target, "raw":raw_doc })+ '\n')
            
            
    
    
