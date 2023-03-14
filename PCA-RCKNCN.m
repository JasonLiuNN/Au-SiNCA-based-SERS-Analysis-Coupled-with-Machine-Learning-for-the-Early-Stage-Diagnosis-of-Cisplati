%================PCA============%
[coeff,score,latent,tsquared,explained,mu] = pca(data);
for i=1:length(explained)
    a(i)=sum(explained(1:i));
    if sum(explained(1:i))>95
        count=i;
        break;
    end
end

    tran=coeff(:,1:count);
   X=data*tran;
    Data=X;

%==============RCKNCN=============%
function [idx,R,trainClass,zl,ZN] = rckncnfunction(trainData,trainClass,testData,K)
lamda = 1 ;
classNum=length(unique(trainClass));
[N,M]=size(trainData);
dist=zeros(N,1);
for i=1:N
    dist(i,:)=norm(trainData(i,:)-testData);
end
[~,I]=sort(dist,1); 
ZN=zeros(K,M);
ZN(1,:)=trainData(I(1),:);
m=zeros(K,M);
label=zeros(1,N);
label(I(1))=1;
zl=zeros(1,K);
zl(1)=I(1);
R=zeros(classNum);
if K>1
for i=2:K
    m(i,:)=m(i-1,:)+ZN(i-1,:);
    D=1e10;  
    for l=1:N
        ZC=(m(i,:)+trainData(l,:))/i;
        d=sqrt(sum((testData-ZC).^2,2));
        if(d<D && label(l)~=1)
            D=d;
            zl(i)=l;
        end    
    end
    ZN(i,:)=trainData(zl(i),:);
    label(zl(i))=1;
end
end
R=(ZN*ZN'+lamda*eye(K))^(-1)*ZN*testData';
trainClass=trainClass(zl);
labels=zeros(1,classNum);
for i=1:K
    labels(trainClass(i))=labels(trainClass(i))+abs(R(i));
end
[~,idx]=max(labels);
end
 


