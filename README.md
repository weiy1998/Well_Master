![image](https://github.com/user-attachments/assets/771bcc87-4cb9-4e4a-b16a-896419b88881)QAnythingV2部署

1.下载文件
2.解压缩QAnything
tar -xvzf QAnything.tar.gz
3.合并镜像压缩包
cat qanything_offline.tar.xz.* > qanything_offline.tar.xz
4.解压缩镜相包
xz -d qanything_offline.tar.xz
得到qanything_offline.tar（Docker镜像文件）
5.导入镜像文件
docker load -i qanything_offline.tar
6.解压缩ollama镜像文件
xz -d ollama.tar.xz
7.导入ollama镜像文件
docker load -i ollama.tar
8.创建ollama容器
docker run -d --name ollama --gpus all --network host ollama:1
9.部署Qwen2.5大模型
docker exec -it ollama /bin/bash  （进入ollama容器）
ollama pull qwen2.5:7b
10.修改模型默认的上下文长度（默认的2048）
ollama show --modelfile qwen2.5:7b > Modelfile
vi Modelfile
在From下面加一行
PARAMETER num_ctx 32000
ollama create -f Modelfile qwen2.5:7b（重新生成模型文件）
ollama run qwen2.5:7b（运行）
/show parameters （查看设置的参数有效性）
11.进入QAnything文件夹启动QAnything
docker compose -f docker-compose-linux.yaml up
启动成功后，用http://10.242.187.60:8777/qanything/ 访问 （10.242.187.60主机ip地址）


后续的修改


容器被关闭后，第二次重启之前的容器
docker start c1219a9d051bf775dbea9f893e3b4e2bd2a3060e6c207bf900bcb223d1c3a56a
进入容器
docker exec -it ollama /bin/bash
运行模型
ollama run qwen2.5:7b

可能模型太大了，CPU一直处于高负载，因此考虑换一下3b模型

ollama pull qwen2.5:3b-instruct-q6_K
修改tokens
ollama show --modelfile qwen2.5:3b-instruct-q6_K > Modelfile
vi Modelfile
在From下面加一行
PARAMETER num_ctx 32000
ollama create -f Modelfile qwen2.5:3b-instruct-q6_K_ctx32k（重新生成模型文件）
ollama run qwen2.5:3b-instruct-q6_K_ctx32k（运行）

降低参数模型后，各个占用都变少很多了。

