use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{ChatCompletionRequestMessage, ChatCompletionResponseMessage, Role},
    Client,
};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod chat_requests {
    use async_openai::{
        config::OpenAIConfig,
        error::OpenAIError,
        types::{
            ChatCompletionRequestMessage, ChatCompletionResponseMessage,
            CreateChatCompletionRequestArgs,
        },
        Client,
    };

    const CHAT_MODEL: &str = "gpt-3.5-turbo";

    /// Asynchronously make a request to `CHAT_MODEL`, returning the Result.
    /// The `"gpt-3.5-turbo` model is used if None is supplied for the model.
    #[tokio::main]
    pub async fn requeset_chat_model(
        client: &Client<OpenAIConfig>,
        messages: Vec<ChatCompletionRequestMessage>,
        model: Option<&str>,
    ) -> Result<ChatCompletionResponseMessage, OpenAIError> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model.unwrap_or(CHAT_MODEL))
            .messages(messages)
            .max_tokens(100_u16)
            .temperature(0.5)
            .build()?;

        let response = client.chat().create(request).await?;

        Ok(response
            .choices
            .first()
            .expect("Response had an empty choice field")
            .message
            .to_owned())
    }
}

pub trait ChatMessageTrait {
    /// Returns a copy the role of this Chat message
    fn get_role(&self) -> Role;

    /// Returns a copy the contents of this chat message.
    /// Reutrns an empty string if there's no content
    fn get_content(&self) -> String;
}

impl ChatMessageTrait for ChatCompletionRequestMessage {
    fn get_role(&self) -> Role {
        self.role.clone()
    }

    fn get_content(&self) -> String {
        match self.content.clone() {
            Some(c) => c,
            None => String::new(),
        }
    }
}

impl ChatMessageTrait for ChatCompletionResponseMessage {
    fn get_role(&self) -> Role {
        self.role.clone()
    }

    fn get_content(&self) -> String {
        match self.content.clone() {
            Some(c) => c,
            None => String::new(),
        }
    }
}

/// Message parsed from a ChatMessageTrait trait object.
/// This is mainly to add extra data to it
///
/// I feel like this is causing more complexity than it's worth
#[derive(Debug, Clone)]
pub struct Message {
    id: usize,
    content: String,
    created_at: u64,
    role: Role,
}

impl Message {
    /// Create a new message with the given `id`, `role`, and `content`.
    fn new(id: usize, role: Role, content: String) -> Message {
        let created_at = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(n) => n.as_secs(),
            Err(_) => 0,
        };

        Message {
            id,
            content,
            role,
            created_at,
        }
    }

    /// Returns a copy of the id of this message
    pub fn get_id(&self) -> usize {
        self.id.clone()
    }

    /// Returns a copy of the contents of this message
    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    /// Returns a copy of the unix timestamp when this message of created.
    pub fn get_created_at(&self) -> u64 {
        self.created_at.clone()
    }

    /// Returns a copy of the role of this message
    pub fn get_role(&self) -> Role {
        self.role.clone()
    }

    /// Parse this message into an async_openai::type::ChatCompletionRequestMessage
    /// The `name` and `function_call` are left as None.
    pub fn to_chat_resquest_msg(&self) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage {
            role: self.role.clone(),
            content: Some(self.content.clone()),
            name: None,
            function_call: None,
        }
    }

    /// Parse this message into an async_openai::type::ChatCompletionRequestMessage
    pub fn to_chat_response_msg(&self) -> ChatCompletionResponseMessage {
        ChatCompletionResponseMessage {
            role: self.role.clone(),
            content: Some(self.content.clone()),
            function_call: None,
        }
    }
}

/// Struct for each individual chat session
#[derive(Debug, Clone)]
pub struct ChatSession {
    /// A unique id number for this chat session
    id: usize,
    /// The title of this session
    title: String,
    /// messages in this session
    messages: Vec<Message>,
    /// Counter for the ids of this session's messages. Ids are
    /// Guaranteed to be unique for each message in this session.
    msg_id_counter: usize,

    ///The chat model being used by this session
    model: String,
}

impl ChatSession {
    /// Create a new Chat session with the supplied id and title
    fn new(id: usize, title: String, model: &str) -> ChatSession {
        ChatSession {
            id,
            title,
            messages: vec![],
            msg_id_counter: 0,
            model: model.to_string(),
        }
    }

    /// Processes a User message and makes a request to
    /// The chat model. If successful, both the message and response
    /// are stored, returning an Ok(()). Otherwise an Err(OpenAIError)
    /// is returned
    pub fn add_message(
        &mut self,
        contents: String,
        client: &Client<OpenAIConfig>,
    ) -> Result<(), OpenAIError> {
        use chat_requests::requeset_chat_model;

        let chat_request_msg = ChatCompletionRequestMessage {
            role: Role::User,
            content: Some(contents.clone()),
            name: None,
            function_call: None,
        };

        let msg = Message::new(self.msg_id_counter.to_owned(), Role::User, contents);

        let mut temp_messages = self.messages.to_vec();
        temp_messages.push(msg);

        let response = requeset_chat_model(
            client,
            temp_messages
                .to_vec()
                .iter()
                .map(|x| x.to_chat_resquest_msg())
                .collect(),
            Some(self.model.as_str()),
        )?;

        self.add_chat_message(chat_request_msg);
        self.add_chat_message(response);

        Ok(())
    }

    /// Adds a new chat message to this session, consuming it in the process.
    ///
    /// Chat Message should implement the ChatMessageTrait. Currently only the
    /// `async_openai::types::{ChatCompletionResponseMessage, ChatCompletionRequestMessage}`
    /// implement this.
    ///
    /// The id and created at fields are automatically added to the message.
    fn add_chat_message<T: ChatMessageTrait>(&mut self, msg: T) {
        let id = self.msg_id_counter;
        self.msg_id_counter += 1;

        self.messages
            .push(Message::new(id, msg.get_role(), msg.get_content()));
    }
    /// Deletes message with matching id in this chat session. The right most,
    /// deleted message is returned if possible.
    pub fn delete_message(&mut self, id: usize) -> Option<Message> {
        // Same general approach like Store.delete_session()
        let mut accumulator: Vec<Message> = Vec::new();

        let target = self.messages.iter().fold(None, |acc, curr| {
            if curr.id == id {
                return Some(curr.to_owned());
            } else {
                accumulator.push(curr.to_owned());
                return acc;
            }
        });
        self.messages = accumulator;

        target
    }

    /// Renames this session to the `new_title` provided. `new_title` is
    /// consumed in the process
    pub fn rename_session(&mut self, new_title: String) {
        self.title = new_title;
    }

    /// Returns a copy of the title of this session
    pub fn get_title(&self) -> String {
        self.title.clone()
    }

    /// Returns a copy of the id of this session
    pub fn get_id(&self) -> usize {
        self.id.clone()
    }
    /// Returns a reference to the collection of messages in this session
    pub fn get_messages(&self) -> &Vec<Message> {
        self.messages.as_ref()
    }
}

/// Struct for storing all Chat sessions
/// Hold functions for creating and keeping track of new sessions
#[derive(Debug, Clone)]
pub struct Store {
    sessions: Vec<ChatSession>,

    /// Counter for the ids of this store's sessions. Ids are
    /// Guaranteed to be unique for each message in this session.
    session_id_counter: usize,

    ///The client for this store. Only supports the
    /// OpenAI REST API based on OpenAPI spec.
    client: Client<OpenAIConfig>,
}

impl Store {
    ///Create a new Store with no sessions. Ownership of the
    /// client is moved to this struct
    pub fn new(client: Client<OpenAIConfig>) -> Store {
        Store {
            sessions: Vec::new(),
            session_id_counter: 0,
            client,
        }
    }

    /// Returns reference to the collection of sessions
    /// in this store
    pub fn get_all_sessions(&self) -> &Vec<ChatSession> {
        self.sessions.as_ref()
    }

    /// Finds and returns a reference to the chat session with matching id if any exists
    pub fn get_session(&self, id: usize) -> Option<&ChatSession> {
        self.sessions.iter().find(|x| x.get_id() == id)
    }

    /// Add a new message to the store, creating a chat session  for it.
    /// `title` is the title of the chat session created. The message is
    /// consumed in the process.
    /// `model` must be a valid chat model
    ///
    /// Only request messages can be used to create a new session
    pub fn add_session(
        &mut self,
        msg: ChatCompletionRequestMessage,
        title: String,
        model: &str,
    ) -> Result<(), OpenAIError> {
        let id = self.session_id_counter;

        let mut chs = ChatSession::new(id, title, model);

        chs.add_message(msg.get_content(), &self.client)?;

        self.session_id_counter += 1;
        self.sessions.push(chs);

        Ok(())
    }

    /// Deletes any chat session with matching id in this store. The right most,
    /// deleted session is returned if possible.
    pub fn delete_session(&mut self, id: usize) -> Option<ChatSession> {
        let mut accumulator: Vec<ChatSession> = Vec::new();

        // iter().filter() yields a new iterator and would not have returned any elements that
        // fail the predicate. So I settled on this approach.
        let target = self.sessions.iter().fold(None, |acc, curr| {
            if curr.get_id() == id {
                return Some(curr.to_owned());
            } else {
                accumulator.push(curr.to_owned());
                return acc;
            }
        });
        self.sessions = accumulator;

        target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionResponseMessage, Role};
    use std::time::{SystemTime, UNIX_EPOCH};

    const MODEL: &str = "gpt-3.5-turbo";

    #[test]
    fn test_create_chat_thread() {
        let id = 2;
        let title = String::from("Something");
        let mut ctd = ChatSession::new(id.clone(), title.clone(), MODEL);

        assert_eq!(ctd.get_id(), id);
        assert_eq!(ctd.get_title(), title);
        assert!(ctd.get_messages().is_empty());
        assert_eq!(ctd.msg_id_counter, 0);

        ctd.rename_session("New title".to_string());
        assert_eq!(ctd.title, "New title".to_string());
    }

    #[test]
    fn test_response_msg() {
        let role = Role::Assistant;
        let content = String::from("Some content");
        let msg = ChatCompletionResponseMessage {
            role: role.clone(),
            content: Some(content.clone()),
            function_call: None,
        };

        assert_eq!(role, msg.get_role());
        assert_eq!(content, msg.get_content());
    }

    #[test]
    fn test_request_msg() {
        let role = Role::System;
        let msg = ChatCompletionRequestMessage {
            role: role.clone(),
            content: None,
            name: None,
            function_call: None,
        };

        assert_eq!(role, msg.get_role());
        assert_eq!("", msg.get_content());
    }

    #[test]
    fn test_create_message() {
        let id = 25;
        let role = Role::User;
        let contents = String::from("Some random content");
        let time_before = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(n) => n.as_secs(),
            Err(_) => panic!("Should not get here"),
        };

        let msg = Message::new(id.clone(), role.clone(), contents.clone());

        assert_eq!(msg.get_id(), id);
        assert_eq!(msg.get_role(), role);
        assert_eq!(msg.get_content(), contents);
        assert_eq!(time_before, msg.get_created_at());
    }

    #[test]
    fn test_session_add_and_remove() {
        let mut chs = ChatSession::new(25, String::from("Testing adding messages"), MODEL);

        for i in 0..6 {
            if i % 2 == 0 {
                let msg = ChatCompletionRequestMessage {
                    role: Role::User,
                    content: Some(String::from("Request chat")),
                    name: None,
                    function_call: None,
                };
                chs.add_chat_message(msg);
            } else {
                let msg = ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content: Some(String::from("Response chat")),
                    function_call: None,
                };
                chs.add_chat_message(msg);
            }
        }

        assert_eq!(6, chs.get_messages().len());
        assert_eq!(
            chs.get_messages()[0].get_content(),
            String::from("Request chat")
        );
        assert_eq!(
            chs.get_messages()[1].get_content(),
            String::from("Response chat")
        );

        assert_eq!(0, chs.delete_message(0).unwrap().get_id());
        assert_eq!(5, chs.get_messages().len());

        assert_eq!(3, chs.delete_message(3).unwrap().get_id());
        assert_eq!(4, chs.get_messages().len());

        assert!(chs.delete_message(0).is_none());
        assert_eq!(4, chs.get_messages().len())
    }

    #[test]
    fn test_new_store() {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);
        let store = Store::new(client);

        assert!(store.get_all_sessions().is_empty());
        assert_eq!(0, store.session_id_counter);
    }

    #[test]
    fn test_store_add_session() {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);
        let mut store = Store::new(client);

        assert!(store.get_all_sessions().is_empty());

        let msg1 = ChatCompletionRequestMessage {
            role: Role::User,
            content: Some(String::from("msg1")),
            name: None,
            function_call: None,
        };
        store
            .add_session(msg1, "Test Message 1".to_string(), MODEL)
            .unwrap();
        assert_eq!(1, store.get_all_sessions().len());
        assert_eq!(0, store.get_all_sessions()[0].get_id());
        assert_eq!(
            "Test Message 1".to_string(),
            store.get_all_sessions()[0].get_title()
        );

        let msg2 = ChatCompletionRequestMessage {
            role: Role::System,
            content: Some(String::from("msg2")),
            name: None,
            function_call: None,
        };
        store
            .add_session(msg2, "Test msg 2".to_string(), MODEL)
            .unwrap();
        assert_eq!(2, store.get_all_sessions().len());
        assert_eq!(1, store.get_all_sessions()[1].get_id());
        assert_eq!(
            String::from("msg2"),
            store.get_all_sessions()[1].get_messages()[0].get_content()
        );
        assert_eq!(
            Role::User,
            store.get_all_sessions()[1].get_messages()[0].get_role()
        );
    }

    #[test]
    fn test_store_get_specific() {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);
        let mut store = Store::new(client);

        let msg1 = ChatCompletionRequestMessage {
            role: Role::User,
            content: Some(String::from("msg1")),
            name: None,
            function_call: None,
        };
        let msg2 = ChatCompletionRequestMessage {
            role: Role::System,
            content: Some(String::from("msg2")),
            name: None,
            function_call: None,
        };
        let msg3 = ChatCompletionRequestMessage {
            role: Role::Assistant,
            content: Some(String::from("msg3")),
            name: None,
            function_call: None,
        };

        store.add_session(msg1, "Tired".to_string(), MODEL).unwrap();
        store.add_session(msg2, "Tired".to_string(), MODEL).unwrap();
        store.add_session(msg3, "Tired".to_string(), MODEL).unwrap();

        assert_eq!(
            store.get_session(2).unwrap().get_title(),
            "Tired".to_string()
        );
        assert!(store.get_session(40).is_none());
        assert_eq!(store.get_all_sessions().len(), 3);
    }

    #[test]
    fn test_store_delete_sessions() {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);
        let mut store = Store::new(client);

        let msg1 = ChatCompletionRequestMessage {
            role: Role::User,
            content: Some(String::from("msg1")),
            name: None,
            function_call: None,
        };
        let msg2 = ChatCompletionRequestMessage {
            role: Role::System,
            content: Some(String::from("msg2")),
            name: None,
            function_call: None,
        };
        let msg3 = ChatCompletionRequestMessage {
            role: Role::Assistant,
            content: Some(String::from("msg3")),
            name: None,
            function_call: None,
        };

        store.add_session(msg1, "One".to_string(), MODEL).unwrap();
        store.add_session(msg2, "Two".to_string(), MODEL).unwrap();
        store.add_session(msg3, "Three".to_string(), MODEL).unwrap();

        assert_eq!(
            store.delete_session(2).unwrap().get_title(),
            "Three".to_string()
        );
        assert_eq!(store.get_all_sessions().len(), 2);
        assert!(store.get_session(2).is_none());
        assert!(store.delete_session(2).is_none());
    }
}
