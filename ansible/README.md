## How-to Guide
### 1. Install prerequisites
```shell
pip install -r requirements.txt
```

### 2. Create your secret file
After creating, please replace mine at `secrets/*.json`


**Note:** Update `state: absent` to destroy the instance

### 3. Run some more complicated playbooks
#### 3.1. Provision the server and firewall rule
    ```shell
    cd playbook
    ansible-playbook create_compute_instance.yaml
    ```

#### 4.2. Install Docker and run the application
After your instance has been started as the folowing image, get the External IP (e.g., `104.198.109.131` as in the example) and replace it in the inventory file

![Compute Engine](./imgs/compute_engine.png)
, and run the following commands:
    
    ```shell
    cd playbook
    ansible-playbook -i ../inventory install_and_run_docker.yml
    ```
, now, you should be able to access your application via `http://104.198.109.131:30000/docs`