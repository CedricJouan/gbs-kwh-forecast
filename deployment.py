
import requests, json, os, shutil


cpd_service_user_id = 'cedric.jouan@ibm.com'
cpd_api_key = "VmnbrHeM32FMTwcRRA8gORzb8ixiRjHoJNY0VdZc"
cpd_url = "https://dse-cpd401-cluster2.cpolab.ibm.com"

resp = requests.post(f'{cpd_url}/icp4d-api/v1/authorize', verify=False, 
                     json={"username": cpd_service_user_id, "api_key": cpd_api_key})
cpd_token = resp.json()['token']

DEPLOYMENT_SPACE_NAME = "kwh_forecast_deployment_space"
FUCNTION_NAME = "kwh_forecast"

wml_client = None
wml_credentials = {
    "token": cpd_token,
    "instance_id": "openshift",
    "url": cpd_url,
    "version": "4.0"
}

from ibm_watson_machine_learning import APIClient
wml_client = APIClient(wml_credentials)

headers = {}
headers["Content-Type"] = "application/json"
headers["Authorization"] = "Bearer " + cpd_token

space_id = None

for space in wml_client.spaces.get_details()['resources'] :
    if space['entity']['name'] == DEPLOYMENT_SPACE_NAME:
        space_id = space['metadata']['id']
        
if space_id == None :
    space_metadata = {
            wml_client.spaces.ConfigurationMetaNames.NAME: DEPLOYMENT_SPACE_NAME,
            wml_client.spaces.ConfigurationMetaNames.DESCRIPTION: DEPLOYMENT_SPACE_NAME,
        }
    space_details = wml_client.spaces.store(meta_props=space_metadata)
    space_id = space_details['metadata']['id']


wml_client.set.default_space(space_id)

def clear_jobs():
    job_details = wml_client.deployments.get_job_details()
    for job in job_details['resources']:
        wml_client.deployments.delete_job(job['metadata']['id'], hard_delete=True)

def clear_deployments():
    # delete previous deployments 
    existing_deployment_list = wml_client.deployments.get_details()
    for existing_deployment in existing_deployment_list['resources']:
        print("Deleting existing deployment with deployment ID {}".format(existing_deployment['metadata']['id']))
        wml_client.deployments.delete(existing_deployment['metadata']['id'])

def clear_functions():
    # delete previous assets
    existing_function_list = wml_client.repository.get_function_details()
    for existing_function in existing_function_list['resources']:
        print("Deleting existing function with function ID {}".format(existing_function['metadata']['id']))
        wml_client.repository.delete(existing_function['metadata']['id'])
        
def clear_data_assets():
    # delete previous assets
    existing_data_asset_list = wml_client.data_assets.get_details()
    for existing_data_asset in existing_data_asset_list['resources']:
        print("Deleting existing data asset with function ID {}".format(existing_data_asset['metadata']['guid']))
        wml_client.data_assets.delete(existing_data_asset['metadata']['guid'])
            
def clear_space(clear_data=False):
    clear_deployments()
    clear_functions()
    if clear_data:
        clear_data_assets()


def upload_assets_to_deployment_space(client, master_folder_path='./source'):
    shutil.make_archive ("master", "zip", master_folder_path)
    meta_props = {
        client.data_assets.ConfigurationMetaNames.NAME: "master.zip", 
        client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "./master.zip"}
    store_data_details = client.data_assets.store(meta_props) 
    print("master.zip downloaded to deployment space") 
    return store_data_details


def store_function(client, function_name, function, software_spec_name="runtime-22.1-py3.9"):
    sofware_spec_uid = client.software_specifications.get_uid_by_name (software_spec_name) 
    metadata = {
        client.repository.FunctionMetaNames.NAME: function_name, 
        client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid, 
        client.repository.FunctionMetaNames.TAGS: []}
    function_details = client.repository.store_function (meta_props=metadata, function=function) 
    function_uid = client.repository.get_function_uid (function_details) 
    print (f"Function uploaded with name {function_name} and function ID {function_uid}") 
    return function_uid

def create_online_deployment(client, function_uid, deployment_name, serving_name):
    online_metadata = {
        client.deployments.ConfigurationMetaNames.NAME: deployment_name,
        client.deployments.ConfigurationMetaNames.ONLINE: {"parameters": {"serving_name": serving_name}}}
    online_deployment_details = client.deployments.create(function_uid, meta_props=online_metadata)
    online_deployment_name = online_deployment_details["metadata"]["name"]
    online_deployment_id = online_deployment_details ["metadata"]["id"]
    print(f"Deployment name: {online_deployment_name}, deployment ID: {online_deployment_id}")
    return online_deployment_details


wrapper_config = {
    "cpd_url":cpd_url,
    "headers":headers,
    "params":{
        "version":"4.0",
        "space_id":space_id}
        }


def wrapper(config = wrapper_config):

    import requests, os, json, shutil, sys

    cpd_url = config['cpd_url']
    headers = config['headers']
    params = config['params']

    def download_data_asset(asset_name, asset_id):
        asset_response = requests.get(f"{cpd_url}/v2/assets/{asset_id}", params=params, headers=headers, verify=False) 
        attachment_id = asset_response.json()["attachments"][0]["id"] 
        response = requests.get( f"{cpd_url}/v2/assets/{asset_id}/attachments/{attachment_id}", params=params, headers=headers, verify=False) 
        attachment_signed_url = response.json()["url"] 
        att_response = requests.get(cpd_url+attachment_signed_url, verify=False) 
        downloaded_asset = att_response.content 
        with open(asset_name, "wb") as f:
            f.write(downloaded_asset)


    data = {"query":"*:*"} 
    resp = requests.post(f"{cpd_url}/v2/asset_types/data_asset/search", json=data, params=params, headers=headers, verify=False) 
    assets_details = resp.json()["results"] 
    for asset in assets_details: 
        if asset["metadata"]["name"] == "master.zip":
            master_asset_id = asset["metadata"]["asset_id"]
    download_data_asset("master.zip", master_asset_id)
    shutil.unpack_archive ("master.zip", "master", "zip")
    os.chdir('./master')

    from master import main
    import pandas as pd

    def score(payload):
        
        df = pd.DataFrame(payload['input_data'][0]['values'], columns = payload['input_data'][0]['fields'])
        output_payload = main(df)

        return output_payload
    return score


if __name__ == "__main__":
    clear_space(clear_data=True)
    upload_assets_to_deployment_space(wml_client)

    function_uid = store_function(wml_client, FUCNTION_NAME, wrapper)
    deployment_details = create_online_deployment(wml_client, function_uid, FUCNTION_NAME+"_dep", FUCNTION_NAME)

    print(deployment_details)
