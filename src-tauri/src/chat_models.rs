use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionResponseMessage, Role};
use std::time::{SystemTime, UNIX_EPOCH};

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
#[derive(Debug, Clone)]
pub struct Message {
    id: usize,
    content: String,
    created_at: u64,
    role: Role,
}

impl Message {
    /// Create a new message with the given `id`, `role`, and `content`.
    #[allow(unused)]
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

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    pub fn get_created_at(&self) -> u64 {
        self.created_at
    }
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
}

impl ChatSession {
    /// Create a new Chat session with the supplied id and title
    fn new(id: usize, title: String) -> ChatSession {
        ChatSession {
            id,
            title,
            messages: vec![],
            msg_id_counter: 0,
        }
    }

    /// Adds a new chat message to this session, consuming it in the process.
    ///
    /// Chat Message should implement the ChatMessageTrait. Currently only the
    /// `async_openai::types::{ChatCompletionResponseMessage, ChatCompletionRequestMessage}`
    /// implement this.
    ///
    /// The id and created at fields are automatically added to the message.
    pub fn add_chat_message<T: ChatMessageTrait>(&mut self, msg: T) {
        let id = self.msg_id_counter;
        self.msg_id_counter += 1;

        self.messages
            .push(Message::new(id, msg.get_role(), msg.get_content()));
    }
    /// Deletes message with matching id in this chat session. The right most,
    /// deleted message is returned if possible.
    pub fn delete_message(&mut self, id: usize) -> Option<Message> {
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

    pub fn rename_session(&mut self, new_title: String) {
        self.title = new_title;
    }

    pub fn get_title(&self) -> String {
        self.title.clone()
    }

    pub fn get_id(&self) -> usize {
        self.id.clone()
    }

    pub fn get_messages(&self) -> &Vec<Message> {
        self.messages.as_ref()
    }
}

/// Struct for storing all Chat sessions
/// Hold functions for creating and keeping track of new sessions
pub struct Store {
    sessions: Vec<ChatSession>,

    /// Counter for the ids of this store's sessions. Ids are
    /// Guaranteed to be unique for each message in this session.
    msg_id_counter: usize,
}

impl Store {
    ///Create a new Store with no sessions
    pub fn new() -> Store {
        Store {
            sessions: Vec::new(),
            msg_id_counter: 0,
        }
    }

    /// Returns all the active sessions
    pub fn get_all_sessions(&self) -> &Vec<ChatSession> {
        self.sessions.as_ref()
    }

    /// Finds an returns a referencce the chat session with matching id if any exists
    pub fn get_session(&self, id: usize) -> Option<&ChatSession> {
        self.sessions.iter().find(|x| x.get_id() == id)
    }

    /// Add a new message to the store, creating a chat session  for it.
    /// `title` is the title of the chat session created. The message is
    /// consumed in the process.
    ///
    /// Only request messages can be used to create a new session
    pub fn add_session(&mut self, msg: ChatCompletionRequestMessage, title: String) {
        let id = self.msg_id_counter;
        self.msg_id_counter += 1;
        let mut chs = ChatSession::new(id, title);
        chs.add_chat_message(msg);
        self.sessions.push(chs);
    }

    /// Deletes chat session with matching id in this store. The right most,
    /// deleted session is returned if possible.
    pub fn delete_session(&mut self, id: usize) -> Option<ChatSession> {
        let mut accumulator: Vec<ChatSession> = Vec::new();

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

    #[test]
    fn test_create_chat_thread() {
        let id = 2;
        let title = String::from("Something");
        let mut ctd = ChatSession::new(id.clone(), title.clone());

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
        let mut chs = ChatSession::new(25, String::from("Testing adding messages"));

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
        let store = Store::new();

        assert!(store.get_all_sessions().is_empty());
        assert_eq!(0, store.msg_id_counter);
    }

    #[test]
    fn test_store_add_session() {
        let mut store = Store::new();

        assert!(store.get_all_sessions().is_empty());

        let msg1 = ChatCompletionRequestMessage {
            role: Role::User,
            content: Some(String::from("msg1")),
            name: None,
            function_call: None,
        };
        store.add_session(msg1, "Test Message 1".to_string());
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
        store.add_session(msg2, "Test msg 2".to_string());
        assert_eq!(2, store.get_all_sessions().len());
        assert_eq!(1, store.get_all_sessions()[1].get_id());
        assert_eq!(
            String::from("msg2"),
            store.get_all_sessions()[1].get_messages()[0].get_content()
        );
        assert_eq!(
            Role::System,
            store.get_all_sessions()[1].get_messages()[0].get_role()
        );
    }

    #[test]
    fn test_store_get_specific() {
        let mut store = Store::new();

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

        store.add_session(msg1, "Tired".to_string());
        store.add_session(msg2, "Tired".to_string());
        store.add_session(msg3, "Tired".to_string());

        assert_eq!(
            store.get_session(2).unwrap().get_title(),
            "Tired".to_string()
        );
        assert!(store.get_session(40).is_none());
    }

    #[test]
    fn test_store_delete_sessions() {
        let mut store = Store::new();

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

        store.add_session(msg1, "One".to_string());
        store.add_session(msg2, "Two".to_string());
        store.add_session(msg3, "Three".to_string());

        assert_eq!(
            store.delete_session(2).unwrap().get_title(),
            "Three".to_string()
        );
        assert_eq!(store.get_all_sessions().len(), 2);
        assert!(store.get_session(2).is_none());
        assert!(store.delete_session(2).is_none());
    }
}
