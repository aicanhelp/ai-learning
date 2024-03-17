下面是AWS Policy中常用的几种类型：

- Identity-based Policy：Identity-based Policy是基于用户、组或角色的AWS Policy。它控制哪些AWS资源可以被这些实体访问，以及实体对资源的操作。
- Resource-based Policy：Resource-based Policy是基于AWS资源的Policy。它控制哪些其他AWS账户或IAM用户、组或角色可以访问特定的AWS资源。
- Organization-based Policy：Organization-based Policy是适用于AWS Organizations的Policy。它控制在AWS Organizations中如何管理、共享和授权AWS资源以及AWS账户。


###  aws iam重要术语】
#### [ARN]

ARN是Amazon Resource Names的缩写，在AWS里，创建的任何资源有其全局唯一的ARN。ARN是一个很重要的概念，它是访问控制可以到达的最小粒度。在使用AWS SDK时，我们也需要ARN来操作对应的资源。

#### [用户（users）]

在AWS里，一个IAM user和unix下的一个用户几乎等价。你可以创建任意数量的用户，为其分配登录AWS management console所需要的密码，以及使用AWS CLI（或其他使用AWS SDK的应用）所需要的密钥。你可以赋予用户管理员的权限，使其能够任意操作AWS的所有服务，也可以依照Principle of least privilege，只授权合适的权限。

注: 这样创建的用户是没有任何权限的，甚至无法登录，可以进一步为用户关联群组，设置密码和密钥.

#### [群组（groups）]

等同于常见的unix group。将一个用户添加到一个群组里，可以自动获得这个群组所具有的权限。在一家小的创业公司里，其AWS账号下可能会建立这些群组：

Admins：拥有全部资源的访问权限
Devs：拥有大部分资源的访问权限，但可能不具备一些关键性的权限，如创建用户
Ops：拥有部署的权限
Stakeholders：拥有只读权限，一般给manager查看信息之用
注: 默认创建的群组没有任何权限，我们还需要为其添加policy.

#### [角色（roles）]

类似于用户，但没有任何访问凭证（密码或者密钥），它一般被赋予某个资源（包括用户），使其临时具备某些权限。比如说一个EC2实例需要访问DynamoDB，我们可以创建一个具有访问DynamoDB权限的角色，允许其被EC2 Service代入（AssumeRule），然后创建EC2的instance-profile使用这个角色。这样，这个EC2实例就可以访问DynamoDB了。当然，这样的权限控制也可以通过在EC2的文件系统里添加AWS配置文件设置某个用户的密钥（AccessKey）来获得，但使用角色更安全更灵活。角色的密钥是动态创建的，更新和失效都无须特别处理。想象一下如果你有成百上千个EC2实例，如果使用某个用户的密钥来访问AWS SDK，那么，只要某台机器的密钥泄漏，这个用户的密钥就不得不手动更新，进而手动更新所有机器的密钥。这是很多使用AWS多年的老手也会犯下的严重错误。

#### [权限（permissions）]

AWS下的权限都通过policy document描述，就是上面我们给出的那个例子。policy是IAM的核心内容.

#### [Identity/Principal]

从资源访问的角度来看，使用 AWS 资源的其实不单单是具体的人，还可能是 Application。所以，AWS 里面的身份，分几种：

- User
- Application
- Federated User
- Role
能在 AWS IAM 控制台里创建的，只有 User 和 Role。而 User 在创建的时候，可以指定它的访问类型。是凭借用户名密码在 Console 登录，还是使用 Access Key ID 及 Secret 通过 API 来访问，还是两者皆可。

要特别注意的是，User 是直接操作 AWS 资源的用户，而不是你自己开发并部署在 AWS 的系统里面的用户。IAM 的 User 是有数量限制的，最多 5000 个。

如果你开发的系统需要操作 AWS 资源，比如说上传文件到 S3，那你需要用的是 Federated User。通过 OpenID Connect（如 Google/Facebook）或者 SAML 2.0（如 Microsoft AD），你的系统用户可以在登录后换取代表某个 AWS Role 的临时 token 来访问 AWS 资源。

#### [Authentication]

访问和使用 AWS 资源有两种方式，一种是通过页面登录，也就是 Console。一种是通过 AWS API，也就是接口，包括 CLI, SDK 或 HTTPS 请求。

IAM User 在 Console 页面登录需要提供 AWS 帐号名，IAM User 名和密码。AWS 帐号名是 AWS 云服务开通时，系统生成的一串数字，或者是你赋予的别名。它其实就是一个多租户系统里面的租户帐号。 AWS 还会为每个帐号提供一个独特的登录链接

而如果是使用 API 访问 AWS，我们是需要用 IAM User 的 Access Key ID 及 Secret 来为这个 HTTP 请求生成签名的
